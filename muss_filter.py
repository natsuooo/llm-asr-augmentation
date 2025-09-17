import os
import re
import random
import pickle
import joblib
import argparse
import multiprocessing
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
TARGET_TOTAL = 60000                # number of candidates gathered after Step 3
MAX_HOURS = 50                      # 50h matches the paper setting
MAX_DURATION_SEC = MAX_HOURS * 3600
MAX_DURATION_MIN = MAX_DURATION_SEC // 60

SAMPLE_SIZE = 500                   # argmax sampling set size (used only where allowed)
BATCH_SIZE = 32                     # GPU batch for PPL

# weights: (TTR : PPL : TERM) = (6 : 3 : 1) as in the paper
ttr_weight = 0.6
ppl_weight = 0.3
term_weight = 0.1

OUTPUT_DIR = "muss_ttr_ppl_term_060301_paper_consistent"

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
base_root = f"/lustre/home/70988567/airport_asr/output/{version}/{dataset}"
output_base_dir = f"/lustre/home/70988567/airport_asr/output/{version}/tts/{dataset}/{OUTPUT_DIR}"
os.makedirs(output_base_dir, exist_ok=True)
audio_dir = f"{output_base_dir}/audio"
os.makedirs(audio_dir, exist_ok=True)

texts_path = f"{base_root}/generated_dedup.txt"
with open(texts_path, encoding="utf-8") as f:
    raw_texts = [s.strip() for s in f.readlines()]
print(f"#raw_texts: {len(raw_texts)}")

term_path = f"{base_root}/terms.txt"
with open(term_path, encoding="utf-8") as f:
    terms = set([line.strip().lower() for line in f.readlines()])
print(f"#domain terms: {len(terms)}")

save_points = [h*3600 for h in range(10, MAX_HOURS+1, 10)]
save_idx = 0

# -------------------------
# Quality guard (language-specific length constraints; simplified)
# -------------------------
ENABLE_QUALITY_GUARD = True
EN_MIN_WORDS = 5
EN_MAX_WORDS = 200
word_re_for_qc = re.compile(r"[A-Za-z']+")

def pass_quality_guard(text: str) -> bool:
    """Lightweight guard to exclude collapsed or ill-formed outputs."""
    if not ENABLE_QUALITY_GUARD:
        return True
    words = word_re_for_qc.findall(text)
    if not (EN_MIN_WORDS <= len(words) <= EN_MAX_WORDS):
        return False
    # heuristic ASCII share (corpora are English in this experiment)
    ascii_ratio = sum(c.isascii() for c in text) / max(1, len(text))
    if ascii_ratio < 0.8:
        letters = re.findall(r"[A-Za-z\s]", text)
        if len(letters) / max(1, len(text)) < 0.6:
            return False
    return True

texts = [t for t in raw_texts if pass_quality_guard(t)]
print(f"#texts after quality guard: {len(texts)}")

# =========================
# Perplexity (GPT-2) with caching
# =========================
ppl_cache_path = f"{base_root}/ppl_cache.pkl"

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
token_cache_path = f"{base_root}/token_term_cache.pkl"

word_re = re.compile(r"[A-Za-z']+")

def process_text_tokens(text):
    """
    Tokenization for TTR/term features.
    - Lowercased, alphabetic tokens only.
    - TTR uses unique tokens.
    - TERM ratio uses occurrence counts (not unique).
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
# Clustering (labels) load
# =========================
print("Step 1: Load clustering results ...")
# Note: we only need labels; the model is not used here but kept for compatibility.
_ = joblib.load(f"{base_root}/kmeans_model_qwen_{N_CLUSTERS}.joblib")
labels = np.load(f"{base_root}/kmeans_labels_qwen_{N_CLUSTERS}.npy")
clusters = [[] for _ in range(N_CLUSTERS)]
for idx, label in enumerate(labels):
    clusters[label].append(idx)

# =========================
# Scoring utilities
# =========================
def compute_raw_components_for_indices(V_set, idx_list):
    """
    Compute raw (unnormalized) components for the given candidate indices.
      TTR_raw(s)   = |Vocab(s) \ V| / |s|
      PPL_raw(s)   = ppl_scores[idx]
      TERM_raw(s)  = term_scores[idx]
    """
    ttr_raw = []
    ppl_raw = []
    term_raw = []
    for idx in idx_list:
        new_types = len(tokens_list[idx] - V_set)
        length = max(1, lens_list[idx])
        ttr_raw.append(new_types / length)
        ppl_raw.append(ppl_scores[idx])
        term_raw.append(term_scores[idx])
    return np.array(ttr_raw, dtype=float), np.array(ppl_raw, dtype=float), np.array(term_raw, dtype=float)

def minmax_params(arr):
    """Return (min, denom) with denom >= 1e-8 to avoid zero-division."""
    arr = np.asarray(arr, dtype=float)
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    denom = max(1e-8, mx - mn)
    return mn, denom

def normalized_scores(ttr_raw, ppl_raw, term_raw):
    """Min-max normalize each component across the provided arrays and combine."""
    t_mn, t_dn = minmax_params(ttr_raw)
    p_mn, p_dn = minmax_params(ppl_raw)
    d_mn, d_dn = minmax_params(term_raw)
    nttr  = (ttr_raw  - t_mn) / t_dn
    nppl  = (ppl_raw  - p_mn) / p_dn
    nterm = (term_raw - d_mn) / d_dn
    return ttr_weight*nttr + ppl_weight*nppl + term_weight*nterm

def normalized_scores_with_given_minmax(ttr_raw, ppl_raw, term_raw,
                                        t_mn, t_dn, p_mn, p_dn, d_mn, d_dn):
    """Combine with externally supplied min/max (useful for sampling argmax)."""
    nttr  = (ttr_raw  - t_mn) / t_dn
    nppl  = (ppl_raw  - p_mn) / p_dn
    nterm = (term_raw - d_mn) / d_dn
    return ttr_weight*nttr + ppl_weight*nppl + term_weight*nterm

# =========================
# Step 2: In-cluster greedy with LOCAL vocabulary sets (parallelizable)
# =========================
print("Step 2: Per-cluster greedy (LOCAL V) with cluster-wide normalization ...")

def select_reps_for_cluster(indices):
    """
    Greedy selection inside a cluster:
    - Maintain local vocabulary V_local per cluster.
    - At each step:
        * compute raw components for ALL remaining indices (cluster-wide),
          derive min/max for normalization from the FULL remaining set,
        * optionally choose argmax among a sampled subset, BUT normalize
          using the full-set min/max (paper-consistent).
    """
    if not indices:
        return []

    rest = list(indices)
    V_local = set()
    taken = []

    # Repeat up to N_CLUSTER_SAMPLES or until cluster empties
    for _ in range(min(N_CLUSTER_SAMPLES, len(rest))):
        # cluster-wide raw components for normalization params
        ttr_raw_full, ppl_raw_full, term_raw_full = compute_raw_components_for_indices(V_local, rest)
        t_mn, t_dn = minmax_params(ttr_raw_full)
        p_mn, p_dn = minmax_params(ppl_raw_full)
        d_mn, d_dn = minmax_params(term_raw_full)

        # candidate subset for argmax search (sampling to keep speed)
        cand = rest if len(rest) <= SAMPLE_SIZE else random.sample(rest, SAMPLE_SIZE)
        ttr_raw_c, ppl_raw_c, term_raw_c = compute_raw_components_for_indices(V_local, cand)
        scores = normalized_scores_with_given_minmax(ttr_raw_c, ppl_raw_c, term_raw_c,
                                                     t_mn, t_dn, p_mn, p_dn, d_mn, d_dn)
        best_pos = int(np.argmax(scores))
        best_idx = cand[best_pos]

        # update local state
        taken.append(best_idx)
        V_local |= tokens_list[best_idx]
        rest.remove(best_idx)

        if not rest:
            break

    return taken

# Run Step2 in parallel over clusters
with multiprocessing.Pool(processes=num_cores) as pool:
    cluster_reps = list(tqdm(pool.imap(select_reps_for_cluster, clusters),
                             total=len(clusters), desc="Cluster reps"))

total_selected_step2 = sum(len(r) for r in cluster_reps if r)
nonempty_clusters = sum(1 for r in cluster_reps if r)
print(f"#clusters: {len(cluster_reps)}, #nonempty clusters: {nonempty_clusters}, "
      f"#selected (step2): {total_selected_step2}")

# =========================
# Step 3: Cluster-level greedy with GLOBAL vocabulary
# (Start with EMPTY V to reflect the staged selection described in the paper)
# =========================
print("Step 3: Cluster-level greedy (GLOBAL V starts empty) ...")

candidate_clusters = [i for i, reps in enumerate(cluster_reps) if reps]
rest_clusters = list(candidate_clusters)
selected_clusters = []

V_global = set()     # start from empty
token_count_global = 0
current_total = 0

def cluster_components_for_scoring(V_set, cluster_idx):
    """Aggregate cluster-level raw components vs current GLOBAL V."""
    reps = cluster_reps[cluster_idx]
    all_tokens = set()
    total_len = 0
    ppl_vals, term_vals = [], []
    for idx in reps:
        all_tokens |= tokens_list[idx]
        total_len += lens_list[idx]
        ppl_vals.append(ppl_scores[idx])
        term_vals.append(term_scores[idx])

    length = max(1, total_len)
    ttr = len(all_tokens - V_set) / length
    ppl_avg = float(np.mean(ppl_vals)) if ppl_vals else 0.0
    term_avg = float(np.mean(term_vals)) if term_vals else 0.0
    return ttr, ppl_avg, term_avg, all_tokens, total_len

with tqdm(desc="Select clusters", total=TARGET_TOTAL) as pbar:
    while rest_clusters and current_total < TARGET_TOTAL:
        # compute raw for all remaining clusters
        raw_ttr, raw_ppl, raw_term = [], [], []
        cache = {}
        for ci in rest_clusters:
            ttr, ppl_avg, term_avg, cl_tokens, cl_len = cluster_components_for_scoring(V_global, ci)
            raw_ttr.append(ttr)
            raw_ppl.append(ppl_avg)
            raw_term.append(term_avg)
            cache[ci] = (cl_tokens, cl_len)

        # normalize across remaining clusters and pick best
        scores = normalized_scores(np.array(raw_ttr), np.array(raw_ppl), np.array(raw_term))
        best_pos = int(np.argmax(scores))
        best_cluster = rest_clusters[best_pos]
        selected_clusters.append(best_cluster)

        # update global state
        cl_tokens, cl_len = cache[best_cluster]
        V_global |= cl_tokens
        token_count_global += cl_len

        # bookkeeping
        current_total += len(cluster_reps[best_cluster])
        rest_clusters.remove(best_cluster)
        pbar.update(len(cluster_reps[best_cluster]))

print(f"#selected clusters: {len(selected_clusters)}, gathered sentences (step3): {current_total}")

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
# - Strict version: normalize across the FULL remaining candidate set each iteration.
# =========================
print("Step 4: Global greedy with TTS generation (strict normalization over full REST) ...")

# fresh GLOBAL V for final greedy (as a new stage)
V_final = set()
token_count_final = 0
final_selected_idxs = []
duration_sec = 0.0
save_texts = []
save_idx = 0
current_min = 0

from kokoro import KPipeline
pipeline = KPipeline(lang_code='a')
speakers = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole",
    "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck"
]

rest = list(set(candidate_indices))
print(f"#candidates: {len(rest)}, #selected clusters: {len(selected_clusters)}")

tts_queue = []
MAX_QUEUE_SIZE = 10
SR = 24000
MAX_UTT_SEC = 30.0  # discard too-long utterances

with tqdm(total=MAX_DURATION_MIN, desc="Accum. audio minutes", unit="min") as pbar:
    while duration_sec < MAX_DURATION_SEC and rest:
        # compute raw for ALL remaining candidates vs current V_final
        ttr_raw_full, ppl_raw_full, term_raw_full = compute_raw_components_for_indices(V_final, rest)
        # normalize across full remaining set
        scores_full = normalized_scores(ttr_raw_full, ppl_raw_full, term_raw_full)
        best_pos = int(np.argmax(scores_full))
        best_idx = rest[best_pos]

        # send to TTS queue
        tts_queue.append((best_idx, texts[best_idx], random.choice(speakers)))
        rest.pop(best_pos)

        # process TTS queue
        if len(tts_queue) >= MAX_QUEUE_SIZE or len(rest) == 0:
            for idx, text_for_tts, speaker in tts_queue:
                try:
                    generator = pipeline([text_for_tts], voice=speaker)
                    exceeded = False
                    for _, _, audio in generator:
                        duration = len(audio) / SR
                        if duration > MAX_UTT_SEC:
                            exceeded = True
                            break

                        # write wav
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
                            # update final global vocabulary AFTER acceptance
                            V_final |= tokens_list[idx]
                            token_count_final += lens_list[idx]
                        break  # one chunk per utt
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

print(f"Reached budget: {int(duration_sec // 60)} min audio, #selected {len(final_selected_idxs)}, |V|={len(V_final)}")

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
    'token_set': list(V_final),
    'token_count': token_count_final
}
step4_result_path = f"{output_base_dir}/step4_result.pkl"
with open(step4_result_path, 'wb') as f:
    pickle.dump(step4_result, f)
print(f"Saved step4 result: {step4_result_path}")

# Compute final stats on the chosen subset
final_ppl  = [ppl_scores[idx] for idx in final_selected_idxs] if final_selected_idxs else []
final_term = [term_scores[idx] for idx in final_selected_idxs] if final_selected_idxs else []
final_ttr  = (len(V_final) / max(1, token_count_final)) if token_count_final else 0.0

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
print(f"|V| (unique words): {len(V_final)}")
print(f"Total token count: {token_count_final}")
print(f"Weights -> TTR: {ttr_weight}, PPL: {ppl_weight}, TERM: {term_weight}")
print("=============================")
