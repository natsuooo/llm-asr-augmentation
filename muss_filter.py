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

OUTPUT_DIR = "muss_ttr_ppl_term_060301_strict"  # reflect strict matching

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
    raw_texts = [s.strip() for s in f.readlines()]
print(f"#raw_texts: {len(raw_texts)}")

term_path = f"/lustre/home/70988567/airport_asr/output/{version}/{dataset}/terms.txt"
with open(term_path, encoding="utf-8") as f:
    terms = set([line.strip().lower() for line in f.readlines()])
print(f"#domain terms: {len(terms)}")

save_points = [h*3600 for h in range(10, MAX_HOURS+1, 10)]
save_idx = 0

# -------------------------
# Quality guard (paper's language-specific length constraints, simplified)
# -------------------------
ENABLE_QUALITY_GUARD = True
EN_MIN_WORDS = 5
EN_MAX_WORDS = 200

basic_char_re = re.compile(r"^[A-Za-z0-9 ,.;:'\"?!\-()/&%$+#*]+$")  # permissive ASCII

def pass_quality_guard(text: str) -> bool:
    if not ENABLE_QUALITY_GUARD:
        return True
    # count words (English assumption for these corpora)
    words = re.findall(r"[A-Za-z']+", text)
    if not (EN_MIN_WORDS <= len(words) <= EN_MAX_WORDS):
        return False
    # allow some punctuation; if text contains non-ASCII letters a lot, we still allow,
    # but we try to catch obviously broken strings (optional).
    ascii_ratio = sum(c.isascii() for c in text) / max(1, len(text))
    if ascii_ratio < 0.8:
        # still allow if it's mostly letters/spaces
        letters = re.findall(r"[A-Za-z\s]", text)
        if len(letters) / max(1, len(text)) < 0.6:
            return False
    return True

texts = [t for t in raw_texts if pass_quality_guard(t)]
print(f"#texts after quality guard: {len(texts)}")

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
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask   = attention_mask[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        flat_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                             shift_labels.view(-1))
        loss = flat_loss.view(input_ids.size(0), -1)
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
        if (not isinstance(token_data, tuple) or len(token_data) != 3):
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
# KMeans clusters
# =========================
print("Step 1: Load clustering results ...")
kmeans = joblib.load(f"/lustre/home/70988567/airport_asr/output/{version}/{dataset}/kmeans_model_qwen_{N_CLUSTERS}.joblib")
labels = np.load(f"/lustre/home/70988567/airport_asr/output/{version}/{dataset}/kmeans_labels_qwen_{N_CLUSTERS}.npy")
clusters = [[] for _ in range(N_CLUSTERS)]
for idx, label in enumerate(labels):
    clusters[label].append(idx)

# =========================
# Scoring utils (strict per paper)
# =========================
def minmax_norm(arr):
    arr = np.asarray(arr, dtype=float)
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    denom = max(1e-8, mx - mn)
    return (arr - mn) / denom, mn, mx

def compute_raw_components_for_indices(global_token_set, idx_list):
    """
    Compute raw (unnormalized) components for a list of candidate indices,
    using the paper's definition:
      TTR_raw(s)   = |Vocab(s) \ V| / |s|
      PPL_raw(s)   = ppl_scores[idx]
      TERM_raw(s)  = term_scores[idx]
    """
    ttr_raw = []
    ppl_raw = []
    term_raw = []
    for idx in idx_list:
        new_types = len(tokens_list[idx] - global_token_set)
        length = max(1, lens_list[idx])
        ttr_raw.append(new_types / length)
        ppl_raw.append(ppl_scores[idx])
        term_raw.append(term_scores[idx])
    return np.array(ttr_raw, dtype=float), np.array(ppl_raw, dtype=float), np.array(term_raw, dtype=float)

def combined_scores_from_raw(ttr_raw, ppl_raw, term_raw):
    nttr, _, _ = minmax_norm(ttr_raw)
    nppl, _, _ = minmax_norm(ppl_raw)
    nterm, _, _ = minmax_norm(term_raw)
    return (ttr_weight * nttr) + (ppl_weight * nppl) + (term_weight * nterm)

# =========================
# Step 2: In-cluster greedy (global V referenced; round-robin)
# =========================
print("Step 2: Per-cluster greedy (TTR + PPL + TERM) with global V ...")

USE_SAMPLING = True  # set False for stricter but slower selection
cluster_reps = [[] for _ in range(N_CLUSTERS)]
cluster_rest = [list(indices) for indices in clusters]
cluster_taken = [0] * N_CLUSTERS

global_token_set = set()
global_token_count = 0  # not used directly in strict TTR term, kept for completeness

total_step2_selected = 0
progress_total_target = sum(min(N_CLUSTER_SAMPLES, len(r)) for r in cluster_rest)

with tqdm(total=progress_total_target, desc="Cluster reps (round-robin)") as pbar:
    # continue until every cluster meets its quota or empties
    while True:
        progressed = False
        for ci in range(N_CLUSTERS):
            rest = cluster_rest[ci]
            if not rest:
                continue
            if cluster_taken[ci] >= min(N_CLUSTER_SAMPLES, len(clusters[ci])):
                continue

            # candidate set for scoring (sampling optional)
            if USE_SAMPLING and len(rest) > SAMPLE_SIZE:
                cand = random.sample(rest, SAMPLE_SIZE)
            else:
                cand = rest

            # compute raw components w.r.t current GLOBAL V
            ttr_raw, ppl_raw, term_raw = compute_raw_components_for_indices(global_token_set, cand)
            scores = combined_scores_from_raw(ttr_raw, ppl_raw, term_raw)
            best_local = int(np.argmax(scores))
            best_idx = cand[best_local]

            # select best
            cluster_reps[ci].append(best_idx)
            rest.remove(best_idx)
            cluster_taken[ci] += 1
            total_step2_selected += 1
            progressed = True

            # update GLOBAL V per selection (strict adherence)
            global_token_set |= tokens_list[best_idx]
            global_token_count += lens_list[best_idx]

            pbar.update(1)

        if not progressed:
            break

nonempty_clusters = sum(1 for r in cluster_reps if r)
print(f"#clusters: {len(cluster_reps)}, #nonempty clusters: {nonempty_clusters}, "
      f"#selected (step2): {sum(len(r) for r in cluster_reps)}")

# =========================
# Step 3: Cluster-level greedy
# =========================
print("Step 3: Cluster-level greedy (TTR + PPL + TERM) ...")

candidate_clusters = [i for i, reps in enumerate(cluster_reps) if reps]
selected_clusters = []
rest_clusters = list(candidate_clusters)

# We'll recompute scores at each iteration; normalization is across current rest_clusters.
def cluster_raw_components(global_token_set, cluster_idx):
    reps = cluster_reps[cluster_idx]
    all_cluster_tokens = set()
    total_cluster_len = 0
    ppl_vals = []
    term_vals = []
    for idx in reps:
        all_cluster_tokens |= tokens_list[idx]
        total_cluster_len += lens_list[idx]
        ppl_vals.append(ppl_scores[idx])
        term_vals.append(term_scores[idx])

    length = max(1, total_cluster_len)
    ttr = len(all_cluster_tokens - global_token_set) / length
    ppl_avg = float(np.mean(ppl_vals)) if ppl_vals else 0.0
    term_avg = float(np.mean(term_vals)) if term_vals else 0.0
    return ttr, ppl_avg, term_avg, all_cluster_tokens, total_cluster_len

global_token_set_step3 = set(global_token_set)  # start from end of step2
global_token_count_step3 = global_token_count

current_total = 0

with tqdm(desc="Select clusters", total=TARGET_TOTAL) as pbar:
    while rest_clusters and current_total < TARGET_TOTAL:
        # compute raw per cluster
        raw_ttr = []
        raw_ppl = []
        raw_term = []
        cluster_cache = {}  # cache tokens & lengths for update after selection
        for ci in rest_clusters:
            ttr, ppl_avg, term_avg, cl_tokens, cl_len = cluster_raw_components(global_token_set_step3, ci)
            raw_ttr.append(ttr)
            raw_ppl.append(ppl_avg)
            raw_term.append(term_avg)
            cluster_cache[ci] = (cl_tokens, cl_len)

        # normalize across all remaining clusters
        scores = combined_scores_from_raw(np.array(raw_ttr), np.array(raw_ppl), np.array(raw_term))

        # pick best cluster
        best_pos = int(np.argmax(scores))
        best_cidx = rest_clusters[best_pos]
        selected_clusters.append(best_cidx)

        reps = cluster_reps[best_cidx]
        cl_tokens, cl_len = cluster_cache[best_cidx]

        # update global state for step3
        global_token_set_step3 |= cl_tokens
        global_token_count_step3 += cl_len

        rest_clusters.remove(best_cidx)
        current_total += len(reps)
        pbar.update(len(reps))

print(f"#selected clusters: {len(selected_clusters)}, selected sentences (step3): {current_total}")

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

token_set = set()   # fresh global V for final selection (paper's final global greedy)
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
        # STRICT: normalize across the full remaining set 'rest' (can be heavy)
        # For performance, you may switch to sampling; here we keep strict by default.
        cand = rest  # full set

        ttr_raw, ppl_raw, term_raw = compute_raw_components_for_indices(token_set, cand)
        scores = combined_scores_from_raw(ttr_raw, ppl_raw, term_raw)

        best_pos = int(np.argmax(scores))
        best_idx = cand[best_pos]

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
                            # update global V after the actual acceptance
                            token_set |= tokens_list[idx]
                            token_count += lens_list[idx]
                        break  # only one chunk per utt
                except Exception as e:
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
final_ttr  = (len(token_set) / max(1, token_count)) if token_count else 0.0

print("\n=== Final selection stats ===")
print(f"TTR (|V|/tokens): {final_ttr:.4f}")
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
