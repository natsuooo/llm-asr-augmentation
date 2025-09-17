# =========================
# Sample code for MUSS filtering with the combined score of TTR, perplexity, and domain-specific term weighting.
# =========================

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
# Devices / resources
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

num_cores = min(24, multiprocessing.cpu_count())
print(f"CPU cores: {num_cores}")

# -------------------------
# Selection / filtering params
# -------------------------
N_CLUSTERS = 1000
N_CLUSTER_SAMPLES = 200
TARGET_TOTAL = 60000                 # gather after Step 3
MAX_HOURS = 100
MAX_DURATION_SEC = MAX_HOURS * 3600
MAX_DURATION_MIN = MAX_DURATION_SEC // 60

SAMPLE_SIZE = 500                    # argmax sampling set size per iteration
BATCH_SIZE = 32                      # GPU batch for PPL

# weights (alpha:beta:gamma) = (TTR:logPPL:TERM) = (0.6 : 0.3 : 0.1)
ttr_weight = 0.6
ppl_weight = 0.3
term_weight = 0.1

# winsorization quantiles
WINS_LOW_Q = 0.05
WINS_UP_Q  = 0.95
EPS = 1e-8

OUTPUT_DIR = "muss_ttr_logppl_term_060301_winsor"

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
print(f"#all texts: {len(raw_texts)}")

term_path = f"{base_root}/terms.txt"
with open(term_path, encoding="utf-8") as f:
    terms = set([line.strip().lower() for line in f.readlines()])
print(f"#domain terms: {len(terms)}")

save_points = [h*3600 for h in range(10, MAX_HOURS+1, 10)]
save_idx = 0

# -------------------------
# Optional quality guard
# -------------------------
ENABLE_QUALITY_GUARD = False
word_re_for_qc = re.compile(r"[A-Za-z0-9'\-]+")

def pass_quality_guard(text: str) -> bool:
    if not ENABLE_QUALITY_GUARD:
        return True
    toks = word_re_for_qc.findall(text)
    return 5 <= len(toks) <= 200

texts = [t for t in raw_texts if pass_quality_guard(t)]
print(f"#texts after QC: {len(texts)}")

# =========================
# Per-token NLL / PPL with caching
# =========================
# We cache PPL but convert to logPPL (= per-token NLL) via np.log(PPL).
ppl_cache_path = f"{base_root}/ppl_cache.pkl"

def compute_ppl_batch(model, tokenizer, batch_texts):
    """Return per-sample perplexity (exp of avg loss)."""
    enc = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask   = attention_mask[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        flat_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = flat_loss.view(input_ids.size(0), -1)
        token_counts = torch.clamp(shift_mask.sum(dim=1), min=1)
        avg_loss = (loss * shift_mask).sum(dim=1) / token_counts  # this is per-token NLL (log-perplexity)
        ppl = torch.exp(avg_loss).cpu().numpy()
    return ppl

def load_or_compute_ppl():
    ppl_scores = None
    if os.path.exists(ppl_cache_path):
        print(f"Load PPL cache: {ppl_cache_path}")
        with open(ppl_cache_path, 'rb') as f:
            ppl_scores = pickle.load(f)
        if not isinstance(ppl_scores, (list, np.ndarray)) or len(ppl_scores) != len(texts):
            print("PPL cache length mismatch. Recomputing...")
            ppl_scores = None

    if ppl_scores is None:
        print("Prepare GPT-2 for PPL...")
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
                        batch_ppl.append(50.0)
                batch_ppl = np.array(batch_ppl, dtype=float)
            ppl_scores.extend(batch_ppl.tolist())

        with open(ppl_cache_path, 'wb') as f:
            pickle.dump(ppl_scores, f)

    ppl_scores = np.asarray(ppl_scores, dtype=float)
    print(f"PPL stats: min={ppl_scores.min():.2f}, max={ppl_scores.max():.2f}, mean={ppl_scores.mean():.2f}")
    return ppl_scores

ppl_scores = load_or_compute_ppl()
# Convert to per-token NLL (logPPL). Higher = harder = better for our maximization term.
logppl_scores = np.log(np.clip(ppl_scores, EPS, None))

# =========================
# Token / term features with caching
# =========================
token_cache_path = f"{base_root}/token_term_cache.pkl"
word_re = re.compile(r"[A-Za-z0-9'\-]+")

def process_text_tokens(text):
    """
    Tokenization for TTR/TERM features.
    - Lowercased alnum tokens (with ' and -).
    - TTR uses unique token types.
    - TERM ratio uses occurrence counts (not unique).
    """
    toks = [t.lower() for t in word_re.findall(text)]
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
        if isinstance(token_data, tuple) and len(token_data) == 3:
            tokens_list, lens_list, term_scores = token_data
            if len(tokens_list) == len(texts) and len(lens_list) == len(texts) and len(term_scores) == len(texts):
                return list(tokens_list), list(lens_list), np.array(term_scores, dtype=float)
        print("Token cache mismatch. Recomputing...")

    print(f"Parallel tokenization ({num_cores} cores) ...")
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
print(f"TERM ratio stats: min={term_scores.min():.4f}, max={term_scores.max():.4f}, mean={term_scores.mean():.4f}")

# =========================
# Clustering (labels) load
# =========================
print("Step 1: Load clustering ...")
_ = joblib.load(f"{base_root}/kmeans_model_qwen_{N_CLUSTERS}.joblib")
labels = np.load(f"{base_root}/kmeans_labels_qwen_{N_CLUSTERS}.npy")
clusters = [[] for _ in range(N_CLUSTERS)]
for idx, label in enumerate(labels):
    clusters[label].append(idx)

# =========================
# Winsorized min–max helpers
# =========================
def winsor_params(values, low_q=WINS_LOW_Q, up_q=WINS_UP_Q):
    """Compute winsorization quantiles on a vector."""
    arr = np.asarray(values, dtype=float)
    ql = float(np.quantile(arr, low_q)) if len(arr) > 0 else 0.0
    qu = float(np.quantile(arr, up_q))  if len(arr) > 0 else 1.0
    denom = max(qu - ql, EPS)
    return ql, qu, denom

def winsor_norm(x, ql, qu, denom):
    """Apply winsorization then min–max normalize."""
    xw = min(max(x, ql), qu)
    return (xw - ql) / denom

# =========================
# Scoring utilities (per definition)
# =========================
def ttr_raw_for_indices(V_set, idx_list):
    """TTR_raw = |new unique types| / |s| for each candidate w.r.t. current V."""
    raws = []
    for idx in idx_list:
        new_types = len(tokens_list[idx] - V_set)
        length = max(1, lens_list[idx])
        raws.append(new_types / length)
    return np.array(raws, dtype=float)

def combined_score(nttr, nlogppl, nterm):
    return ttr_weight * nttr + ppl_weight * nlogppl + term_weight * nterm

# =========================
# Step 2: In-cluster greedy (LOCAL V) with winsorized min–max
# =========================
print("Step 2: In-cluster greedy with winsorized min–max (TTR/logPPL/TERM) ...")

def select_reps_for_cluster(indices):
    if not indices:
        return []

    rest = list(indices)
    V_local = set()
    taken = []

    for _ in range(min(N_CLUSTER_SAMPLES, len(rest))):
        if not rest:
            break

        # Build vectors over the full remaining pool (cluster-level)
        ttr_vec = ttr_raw_for_indices(V_local, rest)
        logppl_vec = np.array([logppl_scores[i] for i in rest], dtype=float)
        term_vec   = np.array([term_scores[i]   for i in rest], dtype=float)

        # Compute winsorization params on each component
        ttr_ql, ttr_qu, ttr_dn       = winsor_params(ttr_vec)
        logppl_ql, logppl_qu, logppl_dn = winsor_params(logppl_vec)
        term_ql, term_qu, term_dn    = winsor_params(term_vec)

        # Score only a sample for speed; normalize using the above params
        cand = rest if len(rest) <= SAMPLE_SIZE else random.sample(rest, SAMPLE_SIZE)

        best_idx = None
        best_score = -1e18
        for idx in cand:
            # raw components
            ttr_raw = len(tokens_list[idx] - V_local) / max(1, lens_list[idx])
            lpp_raw = logppl_scores[idx]
            term_raw = term_scores[idx]

            # winsorized min–max normalization
            nttr    = winsor_norm(ttr_raw,  ttr_ql,    ttr_qu,    ttr_dn)
            nlogppl = winsor_norm(lpp_raw,  logppl_ql, logppl_qu, logppl_dn)
            nterm   = winsor_norm(term_raw, term_ql,   term_qu,   term_dn)

            score = combined_score(nttr, nlogppl, nterm)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            break

        # Accept and update local V
        taken.append(best_idx)
        V_local |= tokens_list[best_idx]
        rest.remove(best_idx)

    return taken

with multiprocessing.Pool(processes=num_cores) as pool:
    cluster_reps = list(tqdm(pool.imap(select_reps_for_cluster, clusters),
                             total=len(clusters), desc="Cluster reps"))

total_selected_step2 = sum(len(r) for r in cluster_reps if r)
nonempty_clusters = sum(1 for r in cluster_reps if r)
print(f"#clusters: {len(cluster_reps)}, #nonempty: {nonempty_clusters}, "
      f"Step2 selected: {total_selected_step2}")

# =========================
# Step 3: Cluster selection greedy (GLOBAL V starts empty)
# =========================
print("Step 3: Cluster-level greedy with winsorized min–max ...")

candidate_clusters = [i for i, reps in enumerate(cluster_reps) if reps]
rest_clusters = list(candidate_clusters)
selected_clusters = []

V_global = set()
token_count_global = 0
current_total = 0

def cluster_components(cluster_idx, V_set):
    """Aggregate cluster-level components."""
    reps = cluster_reps[cluster_idx]
    all_tokens = set()
    total_len = 0
    lpp_vals, term_vals = [], []
    for idx in reps:
        all_tokens |= tokens_list[idx]
        total_len  += lens_list[idx]
        lpp_vals.append(logppl_scores[idx])
        term_vals.append(term_scores[idx])
    length = max(1, total_len)
    ttr = len(all_tokens - V_set) / length
    logppl_avg = float(np.mean(lpp_vals)) if lpp_vals else 0.0
    term_avg   = float(np.mean(term_vals)) if term_vals else 0.0
    return ttr, logppl_avg, term_avg, all_tokens, total_len

with tqdm(desc="Select clusters", total=TARGET_TOTAL) as pbar:
    while rest_clusters and current_total < TARGET_TOTAL:
        # Build vectors over all remaining clusters
        ttr_list, lpp_list, term_list = [], [], []
        cache = {}
        for ci in rest_clusters:
            ttr, lpp_avg, term_avg, cl_tokens, cl_len = cluster_components(ci, V_global)
            ttr_list.append(ttr)
            lpp_list.append(lpp_avg)
            term_list.append(term_avg)
            cache[ci] = (ttr, lpp_avg, term_avg, cl_tokens, cl_len)

        # Winsorization params on cluster-level distributions
        ttr_ql, ttr_qu, ttr_dn       = winsor_params(ttr_list)
        lpp_ql, lpp_qu, lpp_dn       = winsor_params(lpp_list)
        term_ql, term_qu, term_dn    = winsor_params(term_list)

        # Sample subset for argmax (optional)
        cand = rest_clusters if len(rest_clusters) <= SAMPLE_SIZE else random.sample(rest_clusters, SAMPLE_SIZE)

        best_cluster = None
        best_score = -1e18
        for ci in cand:
            ttr, lpp_avg, term_avg, _, _ = cache[ci]
            nttr    = winsor_norm(ttr,     ttr_ql, ttr_qu, ttr_dn)
            nlogppl = winsor_norm(lpp_avg, lpp_ql, lpp_qu, lpp_dn)
            nterm   = winsor_norm(term_avg,term_ql, term_qu, term_dn)
            score = combined_score(nttr, nlogppl, nterm)
            if score > best_score:
                best_score = score
                best_cluster = ci

        # Accept best cluster
        selected_clusters.append(best_cluster)
        _, _, _, cl_tokens, cl_len = cache[best_cluster]
        V_global |= cl_tokens
        token_count_global += cl_len
        current_total += len(cluster_reps[best_cluster])
        rest_clusters.remove(best_cluster)
        pbar.update(len(cluster_reps[best_cluster]))

print(f"#selected clusters: {len(selected_clusters)}, collected sentences (Step3): {current_total}")

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
print(f"Saved Step3 result: {step3_result_path}")

# =========================
# Step 4: Final global greedy (winsorized min–max) with TTS budget
# =========================
print("Step 4: Global greedy with winsorized min–max and TTS generation ...")

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
        # Build vectors over the full remaining pool (sentence-level)
        ttr_vec    = ttr_raw_for_indices(V_final, rest)
        logppl_vec = np.array([logppl_scores[i] for i in rest], dtype=float)
        term_vec   = np.array([term_scores[i]   for i in rest], dtype=float)

        # Winsorization params over current pool
        ttr_ql, ttr_qu, ttr_dn       = winsor_params(ttr_vec)
        lpp_ql, lpp_qu, lpp_dn       = winsor_params(logppl_vec)
        term_ql, term_qu, term_dn    = winsor_params(term_vec)

        # Score only a sample for speed
        cand = rest if len(rest) <= SAMPLE_SIZE else random.sample(rest, SAMPLE_SIZE)

        best_idx = None
        best_score = -1e18
        for idx in cand:
            ttr_raw  = len(tokens_list[idx] - V_final) / max(1, lens_list[idx])
            lpp_raw  = logppl_scores[idx]
            term_raw = term_scores[idx]

            nttr    = winsor_norm(ttr_raw,  ttr_ql, ttr_qu, ttr_dn)
            nlogppl = winsor_norm(lpp_raw,  lpp_ql, lpp_qu, lpp_dn)
            nterm   = winsor_norm(term_raw, term_ql, term_qu, term_dn)

            score = combined_score(nttr, nlogppl, nterm)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            break

        # Queue for TTS
        tts_queue.append((best_idx, texts[best_idx], random.choice(speakers)))
        rest.remove(best_idx)

        # Run TTS in small batches
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
                            # update V after acceptance
                            V_final |= tokens_list[idx]
                            token_count_final += lens_list[idx]
                        break  # one chunk per utt
                except Exception as e:
                    print(f"[WARN] TTS failed idx={idx}: {e}")

            tts_queue = []

            # Periodic save of text list
            while save_idx < len(save_points) and duration_sec >= save_points[save_idx]:
                save_path = f"{output_base_dir}/selected_texts_{int(save_points[save_idx]//3600)}h.tsv"
                with open(save_path, "w", encoding="utf-8") as f:
                    for wav_filename, t in save_texts:
                        f.write(f"{wav_filename}\t{t}\n")
                print(f"== Saved list at {save_points[save_idx]//3600}h: {save_path}")
                save_idx += 1

            if duration_sec >= MAX_DURATION_SEC:
                break

print(f"Done budget: {int(duration_sec // 60)} min audio, selected {len(final_selected_idxs)}, |V|={len(V_final)}")

# =========================
# Final saves & stats
# =========================
all_save_path = f"{output_base_dir}/selected_texts_all.tsv"
with open(all_save_path, "w", encoding="utf-8") as f:
    for wav_filename, t in save_texts:
        f.write(f"{wav_filename}\t{t}\n")
print(f"Saved list: {all_save_path}")

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

# Final stats (for sanity)
final_lpp  = [logppl_scores[idx] for idx in final_selected_idxs] if final_selected_idxs else []
final_term = [term_scores[idx] for idx in final_selected_idxs] if final_selected_idxs else []
final_ttr  = (len(V_final) / max(1, token_count_final)) if token_count_final else 0.0

print("\n=== Final selection stats ===")
print(f"TTR (|V|/tokens): {final_ttr:.4f}")
if final_lpp:
    print(f"logPPL mean: {np.mean(final_lpp):.3f}, min: {np.min(final_lpp):.3f}, max: {np.max(final_lpp):.3f}")
else:
    print("logPPL: N/A")
if final_term:
    print(f"TERM ratio mean: {np.mean(final_term):.4f}, min: {np.min(final_term):.4f}, max: {np.max(final_term):.4f}")
else:
    print("TERM ratio: N/A")
print(f"|V| (unique words): {len(V_final)}")
print(f"Total token count: {token_count_final}")
print(f"Weights -> TTR: {ttr_weight}, logPPL: {ppl_weight}, TERM: {term_weight}")
print("=============================")
