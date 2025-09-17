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
print(f"デバイス: {device}")

num_cores = min(24, multiprocessing.cpu_count())
print(f"並列処理コア数: {num_cores}")

# -------------------------
# Selection / filtering params
# -------------------------
N_CLUSTERS = 1000
N_CLUSTER_SAMPLES = 200
TARGET_TOTAL = 60000                # gather after Step 3
MAX_HOURS = 100
MAX_DURATION_SEC = MAX_HOURS * 3600
MAX_DURATION_MIN = MAX_DURATION_SEC // 60

SAMPLE_SIZE = 500                   # argmax sampling set size
BATCH_SIZE = 32                     # GPU batch for PPL

# weights （ここでは 0.3 : 0.6 : 0.1）
ttr_weight = 0.3
ppl_weight = 0.6
term_weight = 0.1

OUTPUT_DIR = "muss_ttr_ppl_term_030601_minmax_fixed"

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
print(f"全ての文数: {len(raw_texts)}")

term_path = f"{base_root}/terms.txt"
with open(term_path, encoding="utf-8") as f:
    terms = set([line.strip().lower() for line in f.readlines()])
print(f"専門用語数: {len(terms)}")

save_points = [h*3600 for h in range(10, MAX_HOURS+1, 10)]
save_idx = 0

# -------------------------
# Quality guard（任意、簡易）
# -------------------------
ENABLE_QUALITY_GUARD = False  # 必要なら True
word_re_for_qc = re.compile(r"[A-Za-z0-9'\-]+")

def pass_quality_guard(text: str) -> bool:
    if not ENABLE_QUALITY_GUARD:
        return True
    toks = word_re_for_qc.findall(text)
    if not (5 <= len(toks) <= 200):
        return False
    return True

texts = [t for t in raw_texts if pass_quality_guard(t)]
print(f"品質ガード後の文数: {len(texts)}")

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
        print(f"パープレキシティのキャッシュを読み込み: {ppl_cache_path}")
        with open(ppl_cache_path, 'rb') as f:
            ppl_scores = pickle.load(f)
        # integrity check
        if not isinstance(ppl_scores, (list, np.ndarray)) or len(ppl_scores) != len(texts):
            print("PPLキャッシュの長さ不一致。再計算します...")
            ppl_scores = None

    if ppl_scores is None:
        print("パープレキシティ計算モデルの準備...")
        model_name = "gpt2"
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(model_name)
        model.eval().to(device)

        ppl_scores = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="PPL計算"):
            batch_texts = texts[i:i+BATCH_SIZE]
            try:
                batch_ppl = compute_ppl_batch(model, tokenizer, batch_texts)
            except Exception as e:
                print(f"[WARN] batch PPL failed: {e}. 個別計算にフォールバック。")
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
    print(f"パープレキシティ統計: 最小={ppl_scores.min():.2f}, 最大={ppl_scores.max():.2f}, 平均={ppl_scores.mean():.2f}")
    return ppl_scores

ppl_scores = load_or_compute_ppl()

# =========================
# Token / term features with caching
# =========================
token_cache_path = f"{base_root}/token_term_cache.pkl"

word_re = re.compile(r"[A-Za-z0-9'\-]+")

def process_text_tokens(text):
    """
    Tokenization for TTR/term features.
    - Lowercased, alphabetic/num tokens (英数字と'/-含む)。
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
        print(f"トークン/用語キャッシュを読み込み: {token_cache_path}")
        with open(token_cache_path, 'rb') as f:
            token_data = pickle.load(f)
        # integrity check
        if (isinstance(token_data, tuple) and len(token_data) == 3):
            tokens_list, lens_list, term_scores = token_data
            if len(tokens_list) == len(texts) and len(lens_list) == len(texts) and len(term_scores) == len(texts):
                return list(tokens_list), list(lens_list), np.array(term_scores, dtype=float)
        print("トークンキャッシュ不整合。再計算します...")

    print(f"並列トークナイズ（{num_cores}コア） ...")
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
print(f"専門用語比率統計: 最小={term_scores.min():.4f}, 最大={term_scores.max():.4f}, 平均={term_scores.mean():.4f}")

# -------------------------
# グローバル min–max（PPL/TERM）は一度だけ固定
# -------------------------
def safe_minmax(arr):
    arr = np.asarray(arr, dtype=float)
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    dn = max(1e-8, mx - mn)
    return mn, dn

ppl_min, ppl_dn   = safe_minmax(ppl_scores)
term_min, term_dn = safe_minmax(term_scores)

def norm_ppl(val):
    return (val - ppl_min) / ppl_dn

def norm_term(val):
    return (val - term_min) / term_dn

# =========================
# Clustering (labels) load
# =========================
print("Step 1: クラスタリング読み込み ...")
_ = joblib.load(f"{base_root}/kmeans_model_qwen_{N_CLUSTERS}.joblib")
labels = np.load(f"{base_root}/kmeans_labels_qwen_{N_CLUSTERS}.npy")
clusters = [[] for _ in range(N_CLUSTERS)]
for idx, label in enumerate(labels):
    clusters[label].append(idx)

# =========================
# スコア計算ユーティリティ
# =========================
def ttr_raw_for_indices(V_set, idx_list):
    """現Vに対する各インデックスの TTR_raw = |新規タイプ| / 文長"""
    raws = []
    for idx in idx_list:
        new_types = len(tokens_list[idx] - V_set)
        length = max(1, lens_list[idx])
        raws.append(new_types / length)
    return np.array(raws, dtype=float)

def minmax_params_over_rest_ttr(V_set, idx_list):
    """残余プール（またはクラスタ残余）で TTR の min/max を計算（各イテレーションで更新）"""
    ttr_vals = ttr_raw_for_indices(V_set, idx_list)
    mn = float(np.min(ttr_vals))
    mx = float(np.max(ttr_vals))
    dn = max(1e-8, mx - mn)
    return mn, dn

def combined_score_from_components(nttr, nppl, nterm):
    return ttr_weight * nttr + ppl_weight * nppl + term_weight * nterm

# =========================
# Step 2: クラスタ内 greedy（LOCAL V） with 残余TTR正規化
# =========================
print("Step 2: クラスタ内greedy（TTRは残余でmin-max、PPL/TERMはグローバル） ...")

def select_reps_for_cluster(indices):
    if not indices:
        return []

    rest = list(indices)
    V_local = set()
    taken = []

    # 最大 N_CLUSTER_SAMPLES まで
    for _ in range(min(N_CLUSTER_SAMPLES, len(rest))):
        if not rest:
            break

        # TTR min-max は “クラスタ残余 全体” で求める（サンプルではない）
        ttr_mn, ttr_dn = minmax_params_over_rest_ttr(V_local, rest)

        # スコア計算はサンプル（高速化）。ただし正規化は上の全体 min-max を使用
        cand = rest if len(rest) <= SAMPLE_SIZE else random.sample(rest, SAMPLE_SIZE)

        best_idx = None
        best_score = -1e18

        for idx in cand:
            # TTR
            ttr_raw = len(tokens_list[idx] - V_local) / max(1, lens_list[idx])
            nttr = (ttr_raw - ttr_mn) / ttr_dn
            # PPL/TERM（グローバルで固定正規化）
            nppl  = norm_ppl(ppl_scores[idx])
            nterm = norm_term(term_scores[idx])

            score = combined_score_from_components(nttr, nppl, nterm)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            break

        # 受理
        taken.append(best_idx)
        V_local |= tokens_list[best_idx]
        rest.remove(best_idx)

    return taken

with multiprocessing.Pool(processes=num_cores) as pool:
    cluster_reps = list(tqdm(pool.imap(select_reps_for_cluster, clusters),
                             total=len(clusters), desc="Cluster reps"))

total_selected_step2 = sum(len(r) for r in cluster_reps if r)
nonempty_clusters = sum(1 for r in cluster_reps if r)
print(f"クラスタ数: {len(cluster_reps)}, 非空クラスタ数: {nonempty_clusters}, "
      f"Step2選出合計: {total_selected_step2}")

# =========================
# Step 3: クラスタ選抜（GLOBAL V は空から開始）
# =========================
print("Step 3: クラスタ選抜 greedy（TTRは残余クラスタでmin-max、PPL/TERMはグローバル） ...")

candidate_clusters = [i for i, reps in enumerate(cluster_reps) if reps]
rest_clusters = list(candidate_clusters)
selected_clusters = []

V_global = set()
token_count_global = 0
current_total = 0

def cluster_components_for_scoring(cluster_idx, V_set):
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

with tqdm(desc="クラスタ選択", total=TARGET_TOTAL) as pbar:
    while rest_clusters and current_total < TARGET_TOTAL:
        # 残余クラスタで TTR の min-max を算出
        ttr_list = []
        cache = {}
        for ci in rest_clusters:
            ttr, ppl_avg, term_avg, cl_tokens, cl_len = cluster_components_for_scoring(ci, V_global)
            ttr_list.append(ttr)
            cache[ci] = (ttr, ppl_avg, term_avg, cl_tokens, cl_len)
        ttr_mn = float(np.min(ttr_list))
        ttr_dn = max(1e-8, float(np.max(ttr_list)) - ttr_mn)

        # サンプリング上で argmax（正規化は残余全体のmin-max）
        cand = rest_clusters if len(rest_clusters) <= SAMPLE_SIZE else random.sample(rest_clusters, SAMPLE_SIZE)
        best_cluster = None
        best_score = -1e18

        for ci in cand:
            ttr, ppl_avg, term_avg, cl_tokens, cl_len = cache[ci]
            nttr  = (ttr - ttr_mn) / ttr_dn
            nppl  = norm_ppl(ppl_avg)
            nterm = norm_term(term_avg)
            score = combined_score_from_components(nttr, nppl, nterm)
            if score > best_score:
                best_score = score
                best_cluster = ci

        # 受理＆更新
        selected_clusters.append(best_cluster)
        _, _, _, cl_tokens, cl_len = cache[best_cluster]
        V_global |= cl_tokens
        token_count_global += cl_len
        current_total += len(cluster_reps[best_cluster])
        rest_clusters.remove(best_cluster)
        pbar.update(len(cluster_reps[best_cluster]))

print(f"選択クラスタ数: {len(selected_clusters)}, 収集文数（Step3）: {current_total}")

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
print(f"ステップ3の結果を保存: {step3_result_path}")

# =========================
# Step 4: 最終グローバル greedy（TTRは残余候補でmin-max、PPL/TERMはグローバル）
# =========================
print("Step 4: 全体greedy（TTRは残余でmin-max、PPL/TERMはグローバル, サンプリング） ...")

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
print(f"候補文数: {len(rest)}, 選択クラスタ数: {len(selected_clusters)}")

tts_queue = []
MAX_QUEUE_SIZE = 10
SR = 24000
MAX_UTT_SEC = 30.0  # discard too-long utterances

with tqdm(total=MAX_DURATION_MIN, desc="累積音声分", unit="min") as pbar:
    while duration_sec < MAX_DURATION_SEC and rest:
        # TTR の min-max を “残余全文” から計算
        ttr_mn, ttr_dn = minmax_params_over_rest_ttr(V_final, rest)

        # スコア計算はサンプル（ただし正規化は上の min-max）
        cand = rest if len(rest) <= SAMPLE_SIZE else random.sample(rest, SAMPLE_SIZE)

        best_idx = None
        best_score = -1e18
        for idx in cand:
            ttr_raw = len(tokens_list[idx] - V_final) / max(1, lens_list[idx])
            nttr  = (ttr_raw - ttr_mn) / ttr_dn
            nppl  = norm_ppl(ppl_scores[idx])
            nterm = norm_term(term_scores[idx])
            score = combined_score_from_components(nttr, nppl, nterm)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            break

        # キューへ
        tts_queue.append((best_idx, texts[best_idx], random.choice(speakers)))
        rest.remove(best_idx)

        # バッチでTTS実行
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
                            # 受理後に V を更新
                            V_final |= tokens_list[idx]
                            token_count_final += lens_list[idx]
                        break  # one chunk per utt
                except Exception as e:
                    print(f"[WARN] TTS失敗 idx={idx}: {e}")

            tts_queue = []

            # 途中保存
            while save_idx < len(save_points) and duration_sec >= save_points[save_idx]:
                save_path = f"{output_base_dir}/selected_texts_{int(save_points[save_idx]//3600)}h.tsv"
                with open(save_path, "w", encoding="utf-8") as f:
                    for wav_filename, t in save_texts:
                        f.write(f"{wav_filename}\t{t}\n")
                print(f"== {save_points[save_idx]//3600}時間分のテキスト&ファイル名を保存: {save_path}")
                save_idx += 1

            if duration_sec >= MAX_DURATION_SEC:
                break

print(f"終了: 合成音声 {int(duration_sec // 60)} 分, 選択数 {len(final_selected_idxs)}, |V|={len(V_final)}")

# =========================
# Final saves & stats
# =========================
all_save_path = f"{output_base_dir}/selected_texts_all.tsv"
with open(all_save_path, "w", encoding="utf-8") as f:
    for wav_filename, t in save_texts:
        f.write(f"{wav_filename}\t{t}\n")
print(f"Done! 合計選出: {len(final_selected_idxs)}, 累積音声時間: {int(duration_sec // 60)} 分")
print(f"全データを {all_save_path} に保存しました")

step4_result = {
    'final_selected_idxs': final_selected_idxs,
    'duration_sec': duration_sec,
    'token_set': list(V_final),
    'token_count': token_count_final
}
step4_result_path = f"{output_base_dir}/step4_result.pkl"
with open(step4_result_path, 'wb') as f:
    pickle.dump(step4_result, f)
print(f"ステップ4の結果を保存: {step4_result_path}")

# Compute final stats on the chosen subset
final_ppl  = [ppl_scores[idx] for idx in final_selected_idxs] if final_selected_idxs else []
final_term = [term_scores[idx] for idx in final_selected_idxs] if final_selected_idxs else []
final_ttr  = (len(V_final) / max(1, token_count_final)) if token_count_final else 0.0

print("\n=== 最終選択データの統計 ===")
print(f"TTR（|V|/tokens）: {final_ttr:.4f}")
if final_ppl:
    print(f"PPL 平均: {np.mean(final_ppl):.2f}, 最小: {np.min(final_ppl):.2f}, 最大: {np.max(final_ppl):.2f}")
else:
    print("PPL: N/A")
if final_term:
    print(f"専門用語比率 平均: {np.mean(final_term):.4f}, 最小: {np.min(final_term):.4f}, 最大: {np.max(final_term):.4f}")
else:
    print("専門用語比率: N/A")
print(f"|V|（ユニーク語数）: {len(V_final)}")
print(f"総トークン数: {token_count_final}")
print(f"重み -> TTR: {ttr_weight}, PPL: {ppl_weight}, TERM: {term_weight}")
print("=============================")
