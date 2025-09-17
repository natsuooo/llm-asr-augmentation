import os
import re
import random
import pickle
import joblib
import argparse
import multiprocessing
from functools import partial
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm
import soundfile as sf
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# =========================
# Args & basic configuration
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", required=True)
    return parser.parse_args()

args = parse_args()
dataset = args.dataset

# Versioning (example: atcosim uses v5)
version = "v6"
if dataset == "atcosim":
    version = "v5"

# -------------------------
# Reproducibility
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------
# Selection / filtering params
# -------------------------
N_CLUSTERS = 1000
N_CLUSTER_SAMPLES = 200
TARGET_TOTAL = 60000                # total candidates after Step 3
MAX_HOURS = 50                      # 50h matches the paper setting
MAX_DURATION_SEC = MAX_HOURS * 3600
MAX_DURATION_MIN = MAX_DURATION_SEC // 60

SAMPLE_SIZE = 500                   # sampling size for greedy proposals
BATCH_SIZE = 32                     # GPU batch for PPL

# weights: (TTR : PPL : TERM) = (6 : 3 : 1) as in the paper
ttr_weight = 0.6
ppl_weight = 0.3
term_weight = 0.1

OUTPUT_DIR = "muss_ttr_ppl_term_060301"  # reflect weight ratio

# -------------------------
# Devices / resources
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

num_cores = min(24, multiprocessing.cpu_count())
print(f"CPU cores for parallelism: {num_cores}")

# -------------------------
# Paths
# -------------------------
output_base_dir = f"/lustre/home/70988567/airport_asr/output/{version}/tts/{dataset}/{OUTPUT_DIR}"
os.makedirs(output_base_dir, exist_ok=True)
audio_dir = f"{output_base_dir}/audio"
os.makedirs(audio_dir, exist_ok=True)

texts_path = f"/lustre/home/70988567/airport_asr/output/{version}/{dataset}/generated_dedup.txt"
with open(texts_path, encoding="utf-8") as f:
    texts = [s.strip() for s in f.readlines()]
print(f"#texts: {len(texts)}")

term_path = f"/lustre/home/70988567/airport_asr/output/{version}/{dataset}/terms.txt"
with open(term_path, encoding="utf-8") as f:
    terms = set([line.strip().lower() for line in f.readlines()])
print(f"#domain terms: {len(terms)}")

save_points = [h*3600 for h in range(10, MAX_HOURS+1, 10)]
save_idx = 0

# =========================
# Perplexity (GPT-2) with caching
# =========================
ppl_cache_path = f"/lustre/home/70988567/airport_asr/output/{version}/{dataset}/ppl_cache.pkl"

def compute_ppl_batch(model, tokenizer, batch_texts):
    """Compute per-sample perplexity in a batch (higher is 'harder')."""
    enc = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        # manual token-level loss to get per-sample values
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask   = attention_mask[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        flat_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                             shift_labels.view(-1))
        loss = flat_loss.view(input_ids.size(0), -1)
        # masked average loss
        token_counts = torch.clamp(shift_mask.sum(dim=1), min=1)
        avg_loss = (loss * shift_mask).sum(dim=1) / token_counts
        ppl = torch.exp(avg_loss).cpu().numpy()
    return ppl

def load_or_compute_ppl():
    ppl_scores = None
    if os.path.exists(ppl_cache_path):
        print(f"Load PPL cache: {ppl_cache_path}")
        with open(ppl_cache_path, 'rb') as f:
            ppl_scores = pickle.load(f)
        # integrity check
        if not isinstance(ppl_scores, (list, np.ndarray)) or len(ppl_scores) != len(texts):
            print("PPL cache length mismatch. Recomputing PPL...")
            ppl_scores = None

    if ppl_scores is None:
        print("Preparing GPT-2 model for PPL computation...")
        model_name = "gpt2"
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(model_name)
        model.eval().to(device)

        ppl_scores = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Compute PPL"):
            batch_texts = texts[i:i+BATCH_SIZE]
            try:
                batch_ppl = compute_ppl_batch(model, tokenizer, batch_texts)
            except Exception as e:
                # fallback to single example when batch fails
                print(f"[WARN] batch PPL failed: {e}. Fallback to single samples.")
                batch_ppl = []
                for t in batch_texts:
                    try:
                        enc = tokenizer(t, return_tensors="pt")
                        input_ids = enc.input_ids.to(device)
                        with torch.no_grad():
                            out = model(input_ids, labels=input_ids)
                            loss = out.loss
                        batch_ppl.append(float(torch.exp(loss).cpu().numpy()))
                    except Exception:
                        batch_ppl.append(50.0)  # robust default
                batch_ppl = np.array(batch_ppl, dtype=float)

            ppl_scores.extend(batch_ppl.tolist())

        with open(ppl_cache_path, 'wb') as f:
            pickle.dump(ppl_scores, f)

    ppl_scores = np.asarray(ppl_scores, dtype=float)
    print(f"PPL stats: min={ppl_scores.min():.2f}, max={ppl_scores.max():.2f}, mean={ppl_scores.mean():.2f}")
    return ppl_scores

ppl_scores = load_or_compute_ppl()

# =========================
# Token / term features with caching
# =========================
token_cache_path = f"/lustre/home/70988567/airport_asr/output/{version}/{dataset}/token_term_cache.pkl"

word_re = re.compile(r"[A-Za-z']+")

def process_text_tokens(text):
    """
    Tokenization for TTR/term features.
    - Lowercased, alphabetic tokens only.
    - TTR uses unique tokens.
    - Term ratio uses occurrence counts (not unique).
    """
    toks = word_re.findall(text.lower())
    length = len(toks)
    term_count = sum(1 for tok in toks if tok in terms)
    tokens = set(toks)
    term_ratio = (term_count / length) if length else 0.0
    return tokens, length, term_ratio

def load_or_compute_token_term():
    if os.path.exists(token_cache_path):
        print(f"Load token/term cache: {token_cache_path}")
        with open(token_cache_path, 'rb') as f:
            token_data = pickle.load(f)
        # integrity check
        if (not isinstance(token_data, tuple) or
            len(token_data) != 3):
            print("Token cache format mismatch. Recomputing tokens...")
        else:
            tokens_list, lens_list, term_scores = token_data
            if len(tokens_list) == len(texts) and len(lens_list) == len(texts) and len(term_scores) == len(texts):
                return list(tokens_list), list(lens_list), np.array(term_scores, dtype=float)
            print("Token cache length mismatch. Recomputing tokens...")

    print(f"Parallel tokenization on {num_cores} cores ...")
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap(process_text_tokens, texts),
                            total=len(texts), desc="Tokenize/terms"))
    tokens_list, lens_list, term_scores = zip(*results)
    tokens_list = list(tokens_list)
    lens_list = list(lens_list)
    term_scores = np.array(term_scores, dtype=float)

    with open(token_cache_path, 'wb') as f:
        pickle.dump((tokens_list, lens_list, term_scores), f)
    return tokens_list, lens_list, term_scores

tokens_list, lens_list, term_scores = load_or_compute_token_term()
print(f"Term ratio stats: min={term_scores.min():.4f}, max={term_scores.max():.4f}, mean={term_scores.mean():.4f}")

# =========================
# Estimate TTR delta range (for normalization)
# =========================
print("Estimating TTR delta range ...")
ttr_deltas = []
# To reduce bias, periodically reset the base token set
RESET_INTERVAL = 50
token_set = set()
token_count = 0

for i in range(2000):  # more samples for a stabler range
    if i % RESET_INTERVAL == 0:
        token_set = set()
        token_count = 0

    idx = random.randrange(len(texts))
    cand_tokens = tokens_list[idx]
    cand_len = lens_list[idx]

    current_ttr = (len(token_set) / token_count) if token_count else 0.0
    new_token_set = token_set | cand_tokens
    new_token_count = token_count + cand_len if cand_len else token_count
    new_ttr = (len(new_token_set) / new_token_count) if new_token_count else 0.0
    ttr_deltas.append(new_ttr - current_ttr)

    # advance the state to mimic accumulation
    token_set = new_token_set
    token_count = new_token_count

if len(ttr_deltas) == 0:
    print("[WARN] No TTR samples. Using defaults.")
    ttr_delta_min, ttr_delta_max = -0.1, 0.1
else:
    ttr_delta_min = float(np.min(ttr_deltas))
    ttr_delta_max = float(np.max(ttr_deltas))
print(f"TTR delta stats: min={ttr_delta_min:.4f}, max={ttr_delta_max:.4f}, mean={np.mean(ttr_deltas):.4f}")

# prepare ranges (avoid zero division)
ppl_min, ppl_max = float(np.min(ppl_scores)), float(np.max(ppl_scores))
ppl_range = max(1e-8, ppl_max - ppl_min)
ttr_range = max(1e-8, ttr_delta_max - ttr_delta_min)

# =========================
# KMeans clusters
# =========================
print("Step 1: Load clustering results ...")
kmeans = joblib.load(f"/lustre/home/70988567/airport_asr/output/{version}/{dataset}/kmeans_model_qwen_{N_CLUSTERS}.joblib")
labels = np.load(f"/lustre/home/70988567/airport_asr/output/{version}/{dataset}/kmeans_labels_qwen_{N_CLUSTERS}.npy")
clusters = [[] for _ in range(N_CLUSTERS)]
for idx, label in enumerate(labels):
    clusters[label].append(idx)

# =========================
# Scoring functions
# =========================
def calc_combined_score(token_set, token_count, idx):
    """Compute normalized combined score (higher is better)."""
    new_tokens = tokens_list[idx]
    new_count  = lens_list[idx]

    # ΔTTR (handle initial state properly)
    if token_count == 0:
        current_ttr = 0.0
        new_ttr = (len(new_tokens) / max(1, new_count))
        ttr_delta = new_ttr - current_ttr
    else:
        current_ttr = len(token_set) / token_count
        new_ttr = len(token_set | new_tokens) / (token_count + new_count)
        ttr_delta = new_ttr - current_ttr

    nttr  = (ttr_delta - ttr_delta_min) / ttr_range
    nppl  = (ppl_scores[idx] - ppl_min) / ppl_range
    nterm = term_scores[idx]  # already ratio-like

    return ttr_weight*nttr + ppl_weight*nppl + term_weight*nterm

def update_token_set(token_set, token_count, idx):
    new_tokens = tokens_list[idx]
    new_count  = lens_list[idx]
    return (token_set | new_tokens), (token_count + new_count)

# =========================
# Step 2: In-cluster greedy
# =========================
print("Step 2: Per-cluster greedy selection (TTR + PPL + TERM) ...")

def process_cluster(indices):
    if not indices:
        return []
    selected = []
    rest = list(indices)
    token_set = set()
    token_count = 0

    for _ in range(min(N_CLUSTER_SAMPLES, len(indices))):
        if not rest:
            break
        sample_rest = random.sample(rest, SAMPLE_SIZE) if len(rest) > SAMPLE_SIZE else rest

        best_score, best_idx = -1e18, None
        for idx in sample_rest:
            s = calc_combined_score(token_set, token_count, idx)
            if s > best_score:
                best_score, best_idx = s, idx

        if best_idx is None:
            break
        selected.append(best_idx)
        token_set, token_count = update_token_set(token_set, token_count, best_idx)
        rest.remove(best_idx)

    return selected

with multiprocessing.Pool(processes=num_cores) as pool:
    cluster_reps = list(tqdm(pool.imap(process_cluster, clusters),
                             total=len(clusters), desc="Cluster reps"))

total_selected = sum(len(reps) for reps in cluster_reps if reps)
print(f"#clusters: {len(cluster_reps)}, #nonempty clusters: {sum(1 for r in cluster_reps if r)}, "
      f"#selected (step2): {total_selected}")

# =========================
# Step 3: Cluster-level greedy
# =========================
print("Step 3: Cluster-level greedy (TTR + PPL + TERM) ...")
candidate_clusters = [i for i, reps in enumerate(cluster_reps) if reps]
selected_clusters = []
rest_clusters = list(candidate_clusters)
token_set = set()
token_count = 0
current_total = 0

def calc_cluster_combined_score(token_set, token_count, cluster_idx):
    reps = cluster_reps[cluster_idx]
    if not reps:
        return -1e18, token_set, token_count

    all_cluster_tokens = set()
    total_cluster_len = 0
    for idx in reps:
        all_cluster_tokens |= tokens_list[idx]
        total_cluster_len += lens_list[idx]

    # ΔTTR
    if token_count == 0:
        current_ttr = 0.0
        new_ttr = len(all_cluster_tokens) / max(1, total_cluster_len)
        ttr_delta = new_ttr - current_ttr
    else:
        current_ttr = len(token_set) / token_count
        new_ttr = len(token_set | all_cluster_tokens) / (token_count + total_cluster_len)
        ttr_delta = new_ttr - current_ttr

    nttr = (ttr_delta - ttr_delta_min) / ttr_range

    cluster_ppl_avg = float(np.mean([ppl_scores[idx] for idx in reps]))
    nppl = (cluster_ppl_avg - ppl_min) / ppl_range

    cluster_term_avg = float(np.mean([term_scores[idx] for idx in reps]))
    nterm = cluster_term_avg

    combined = ttr_weight*nttr + ppl_weight*nppl + term_weight*nterm
    new_token_set = token_set | all_cluster_tokens
    new_token_count = token_count + total_cluster_len
    return combined, new_token_set, new_token_count

with tqdm(desc="Select clusters", total=TARGET_TOTAL) as pbar:
    while rest_clusters and current_total < TARGET_TOTAL:
        sample_clusters = random.sample(rest_clusters, SAMPLE_SIZE) if len(rest_clusters) > SAMPLE_SIZE else rest_clusters

        best_score, best_cidx = -1e18, None
        best_new_token_set, best_new_token_count = None, None

        for ci in sample_clusters:
            score, new_tset, new_tcnt = calc_cluster_combined_score(token_set, token_count, ci)
            if score > best_score:
                best_score = score
                best_cidx = ci
                best_new_token_set = new_tset
                best_new_token_count = new_tcnt

        if best_cidx is None:
            break

        selected_clusters.append(best_cidx)
        reps = cluster_reps[best_cidx]
        token_set = best_new_token_set
        token_count = best_new_token_count
        rest_clusters.remove(best_cidx)
        current_total += len(reps)
        pbar.update(len(reps))

print(f"#selected clusters: {len(selected_clusters)}, |V|={len(token_set)}, selected sentences (step3): {current_total}")

candidate_indices = []
for ci in selected_clusters:
    candidate_indices += cluster_reps[ci]

step3_result = {
    'selected_clusters': selected_clusters,
    'candidate_indices': candidate_indices,
    'texts': texts,
    'ppl_scores': ppl_scores.tolist(),
    'term_scores': term_scores.tolist()
}
step3_result_path = f"{output_base_dir}/step3_result.pkl"
with open(step3_result_path, 'wb') as f:
    pickle.dump(step3_result, f)
print(f"Saved step3 result: {step3_result_path}")

# =========================
# Step 4: Global greedy with TTS budget
# =========================
print("Step 4: Global greedy with TTS generation (TTR + PPL + TERM) ...")

token_set = set()
token_count = 0
final_selected_idxs = []
duration_sec = 0.0
save_texts = []
save_idx = 0
current_min = 0

from kokoro import KPipeline
pipeline = KPipeline(lang_code='a')
speakers = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck"
]

rest = list(set(candidate_indices))
print(f"#candidates: {len(rest)}, #selected clusters: {len(selected_clusters)}")

tts_queue = []
MAX_QUEUE_SIZE = 10
SR = 24000
MAX_UTT_SEC = 30.0  # discard too-long utterances

with tqdm(total=MAX_DURATION_MIN, desc="Accum. audio minutes", unit="min") as pbar:
    while duration_sec < MAX_DURATION_SEC and rest:
        sample_rest = random.sample(rest, SAMPLE_SIZE) if len(rest) > SAMPLE_SIZE else rest

        best_score, best_idx = -1e18, None
        for idx in sample_rest:
            s = calc_combined_score(token_set, token_count, idx)
            if s > best_score:
                best_score, best_idx = s, idx

        if best_idx is None:
            break

        tts_queue.append((best_idx, texts[best_idx], random.choice(speakers)))
        rest.remove(best_idx)

        if len(tts_queue) >= MAX_QUEUE_SIZE or len(rest) == 0:
            # process queue
            for idx, text_for_tts, speaker in tts_queue:
                try:
                    generator = pipeline([text_for_tts], voice=speaker)
                    exceeded = False
                    for _, _, audio in generator:
                        duration = len(audio) / SR
                        if duration > MAX_UTT_SEC:
                            exceeded = True
                            break

                        # save audio
                        duration_sec += duration
                        new_min = int(duration_sec // 60)
                        if new_min > current_min:
                            pbar.update(new_min - current_min)
                            current_min = new_min

                        wav_filename = f"{str(len(save_texts)).zfill(5)}.wav"
                        sf.write(f"{audio_dir}/{wav_filename}", audio, SR)
                        save_texts.append((wav_filename, text_for_tts))

                        if not exceeded:
                            final_selected_idxs.append(idx)
                            token_set, token_count = update_token_set(token_set, token_count, idx)
                        break  # only one chunk per utt
                except Exception as e:
                    # robust to TTS failures
                    print(f"[WARN] TTS failed for idx={idx}: {e}")

            tts_queue = []

            # periodic save of text list
            while save_idx < len(save_points) and duration_sec >= save_points[save_idx]:
                save_path = f"{output_base_dir}/selected_texts_{int(save_points[save_idx]//3600)}h.tsv"
                with open(save_path, "w", encoding="utf-8") as f:
                    for wav_filename, t in save_texts:
                        f.write(f"{wav_filename}\t{t}\n")
                print(f"== Saved text list at {save_points[save_idx]//3600}h: {save_path}")
                save_idx += 1

            if duration_sec >= MAX_DURATION_SEC:
                break

print(f"Reached budget: {int(duration_sec // 60)} min audio, #selected {len(final_selected_idxs)}, |V|={len(token_set)}")

# =========================
# Final saves & stats
# =========================
all_save_path = f"{output_base_dir}/selected_texts_all.tsv"
with open(all_save_path, "w", encoding="utf-8") as f:
    for wav_filename, t in save_texts:
        f.write(f"{wav_filename}\t{t}\n")
print(f"Done! total selected: {len(final_selected_idxs)}, total minutes: {int(duration_sec // 60)}")
print(f"Saved: {all_save_path}")

step4_result = {
    'final_selected_idxs': final_selected_idxs,
    'duration_sec': duration_sec,
    'token_set': list(token_set),
    'token_count': token_count
}
step4_result_path = f"{output_base_dir}/step4_result.pkl"
with open(step4_result_path, 'wb') as f:
    pickle.dump(step4_result, f)
print(f"Saved step4 result: {step4_result_path}")

# Compute final stats on the chosen subset
final_ppl  = [ppl_scores[idx] for idx in final_selected_idxs] if final_selected_idxs else []
final_term = [term_scores[idx] for idx in final_selected_idxs] if final_selected_idxs else []
final_ttr  = (len(token_set) / token_count) if token_count else 0.0

print("\n=== Final selection stats ===")
print(f"TTR: {final_ttr:.4f}")
if final_ppl:
    print(f"PPL mean: {np.mean(final_ppl):.2f}, min: {np.min(final_ppl):.2f}, max: {np.max(final_ppl):.2f}")
else:
    print("PPL: N/A")
if final_term:
    print(f"Term ratio mean: {np.mean(final_term):.4f}, min: {np.min(final_term):.4f}, max: {np.max(final_term):.4f}")
else:
    print("Term ratio: N/A")
print(f"|V| (unique words): {len(token_set)}")
print(f"Total token count: {token_count}")
print(f"Weights -> TTR: {ttr_weight}, PPL: {ppl_weight}, TERM: {term_weight}")
print("=============================")
