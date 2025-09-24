import os
import json
import gc
import re
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import nltk
from nltk.corpus import wordnet as wn, brown, stopwords

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from sklearn.decomposition import PCA

from concept_labeler import (
    load_embedder,
    build_label_index,
    label_concepts_with_dictionary,
    load_reranker,
)

# 0) Setup & HF models

device = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')  



# Generator (for stimuli)

GEN_NAME = "./llama2-7b-local"

BF16_OK = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
GEN_DTYPE = torch.bfloat16 if BF16_OK else (torch.float16 if torch.cuda.is_available() else torch.float32)

gen_tok = AutoTokenizer.from_pretrained(GEN_NAME)
gen_mdl = AutoModelForCausalLM.from_pretrained(
    GEN_NAME,
    device_map="auto",
    torch_dtype=GEN_DTYPE,
)
dev = next(gen_mdl.parameters()).device
print("GEN device:", dev, "| dtype:", GEN_DTYPE)

if gen_tok.pad_token_id is None and gen_tok.eos_token_id is not None:
    gen_tok.pad_token_id = gen_tok.eos_token_id
gen_tok.padding_side = "left"


# single SBERT embedder instance (reuse for both label index + label vectors)
embedder = load_embedder("sentence-transformers/all-mpnet-base-v2")


# 1) Load intermediate results

with open("prompts.json", "r") as f:
    prompts = json.load(f)

with open("concept_to_prompts.json", "r") as f:
    concept_to_prompts = json.load(f)

# Keep int keys and drop empties
concept_to_prompts = {int(k): v for k, v in concept_to_prompts.items() if (len(v) > 0)}
concept_ids = list(concept_to_prompts.keys())

print("concept_to_prompts (non-empty):", len(concept_to_prompts))


# 2) Build candidate label dictionary (WordNet/Brown/ConceptNet + corpus prior)

print("Loading ConceptNet...")
df = pd.read_csv("conceptnet-assertions-5.7.0.csv.gz", sep="\t", header=None, compression="gzip")
df.columns = ["uri", "relation", "source", "target", "data"]

def _uri_to_text(uri: str) -> str:
    parts = uri.split("/")
    return parts[-1].replace("_", " ")

mask_source = df["source"].str.startswith("/c/en/")
mask_target = df["target"].str.startswith("/c/en/")
sources = df.loc[mask_source, "source"].map(_uri_to_text)
targets = df.loc[mask_target, "target"].map(_uri_to_text)

def clean_label(s: str) -> str:
    s = s.strip().lower().replace("’", "'")
    s = re.sub(r"\s+", " ", s)
    toks = s.split()
    if not (1 <= len(toks) <= 5):
        return ""
    if any(not re.fullmatch(r"[a-z]+(?:[-'][a-z]+)?", t) for t in toks):
        return ""
    bad = ("-like", "-ish")
    if any(t.endswith(bad) for t in toks):
        return ""
    return " ".join(toks)

conceptnet_set = {clean_label(x) for x in list(sources) + list(targets)}
conceptnet_set = {w for w in conceptnet_set if w}
del df, sources, targets
gc.collect()

nltk.download("wordnet"); nltk.download("brown"); nltk.download("punkt"); nltk.download("stopwords")
wordnet_set = {clean_label(synset.name().split(".")[0]) for synset in wn.all_synsets()}
wordnet_set = {w for w in wordnet_set if w}

EN_STOPS = set(stopwords.words("english"))

def normalize_token(tok: str) -> str:
    tok = tok.strip().lower().replace("’", "'")
    tok = re.sub(r"^[^a-z]+|[^a-z]+$", "", tok)
    if not tok: return ""
    if not re.fullmatch(r"[a-z]+([\-'][a-z]+)*", tok): return ""
    return tok

def brown_top_words(n_top: int = 40000, min_len: int = 2, drop_stops: bool = True) -> list[str]:
    cnt = Counter()
    for sent in brown.sents():
        for w in sent:
            w = normalize_token(w)
            if not w: continue
            if len(w) < min_len: continue
            if drop_stops and w in EN_STOPS: continue
            cnt[w] += 1
    return [w for w, _ in cnt.most_common(n_top)]

brown_set = set(brown_top_words(n_top=100000))

def corpus_vocab_counts(texts):
    cnt = Counter()
    for t in texts:
        toks = re.findall(r"[a-z]+", t.lower())
        cnt.update(toks)
    return cnt

_cnt = corpus_vocab_counts(prompts)

def in_corpus(label: str, min_count: int = 1) -> bool:
    return any(_cnt.get(tok, 0) >= min_count for tok in label.split())

multiword_wordnet = {w for w in wordnet_set if " " in w}
candidate_pool = (wordnet_set | brown_set) | (conceptnet_set | multiword_wordnet)

dictionary_texts = sorted({w for w in (clean_label(x) for x in candidate_pool) if w and in_corpus(w)})

label_index = build_label_index(dictionary_texts, embedder, use_faiss=True, batch_size=16)
del dictionary_texts, wordnet_set, conceptnet_set, brown_set, EN_STOPS
gc.collect(); torch.cuda.empty_cache()


# 3) Helpers



class SafeLogitsProcessor(LogitsProcessor):
    #clean logits to avoid NaN/Inf in sampling
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Replace NaN/Inf, clamp to a reasonable window to avoid overflow
        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        scores = torch.clamp(scores, min=-1e4, max=1e4)
        return scores

def compute_last_token_activations(prompts_list, model, tokenizer, device, layer_indices, batch_size=16, max_length=128):
    model.eval()
    n_layers = len(layer_indices)
    acc = [[] for _ in range(n_layers)]
    for i in range(0, len(prompts_list), batch_size):
        batch = prompts_list[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        for li, layer_idx in enumerate(layer_indices):
            arr = hidden_states[layer_idx][:, -1, :].detach().cpu().numpy()
            acc[li].append(arr)
        del inputs, outputs, hidden_states
        torch.cuda.empty_cache()
    for li in range(n_layers):
        acc[li] = np.vstack(acc[li]) if acc[li] else np.zeros((0, model.config.hidden_size), dtype=np.float32)
    return acc

def make_template(label, stimulus):
    return f"Consider the <concept {label}> in the following scenario:\nScenario: {stimulus}\nAnswer:"

#  list parsing 
NUM_ITEM_RE = re.compile(
    r'^[\s"“”\'`]*(\d+)[\.\)\-]\s+(.*?)(?=^[\s"“”\'`]*\d+[\.\)\-]\s+|^\s*[-•]\s+|$\Z)',
    flags=re.MULTILINE | re.DOTALL,
)
BULLET_RE = re.compile(r'^[\s"“”\'`]*[-•]\s+(.*)$', flags=re.MULTILINE)

def _clean_line(s: str) -> str:
    s = s.strip().strip(' "\'“”‘’`').rstrip(' ,;')
    s = re.sub(r'^[\s"“”\'`]*\d+[\.\)\-]\s+', '', s)
    s = re.sub(r'^[\s"“”\'`]*[-•]\s+', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def parse_numbered_or_bulleted(text: str, n: int) -> list[str]:
    items = []
    for m in NUM_ITEM_RE.finditer(text):
        body = _clean_line(m.group(2))
        if 5 <= len(body) <= 180:
            items.append(body)
        if len(items) >= n:
            break
    if len(items) < n:
        for m in BULLET_RE.finditer(text):
            body = _clean_line(m.group(1))
            if 5 <= len(body) <= 180:
                items.append(body)
            if len(items) >= n:
                break
    if len(items) < n:
        for ln in re.split(r'\n+', text):
            ln = _clean_line(ln)
            if not ln or len(ln) < 5:
                continue
            if re.search(r'^(concept|task|note|meta|instruction)\b', ln, flags=re.I):
                continue
            items.append(ln)
            if len(items) >= n:
                break
    seen_lower, outs = set(), []
    for s in items:
        key = s.lower()
        if key not in seen_lower:
            outs.append(s); seen_lower.add(key)
        if len(outs) >= n:
            break
    return outs

def _normalize_label_for_prompt(label: str) -> str:
    if label.islower():
        return f"{label} (treat as a general concept; not a specific brand/person/place)"
    gloss_map = {
        "vicissitudes": "ups and downs; sudden changes in circumstances",
    }
    if label.lower() in gloss_map:
        return f"{label} ({gloss_map[label.lower()]})"
    return label

def _post_filter_stimuli(label: str, items: list[str], max_words: int = 20) -> list[str]:
    out, seen = [], set()
    bad_piece = re.compile(rf"\b{re.escape(label)}\w+\b", flags=re.IGNORECASE)
    whole_label = re.compile(rf"\b{re.escape(label)}\b", flags=re.IGNORECASE)
    for s in items:
        t = s.strip().rstrip(",;:")
        if not (5 <= len(t) <= 180):
            continue
        if t.endswith((" and", " with", " of", " to", " for", " but", " or")):
            continue
        if bad_piece.search(t) and not whole_label.search(t):
            continue
        if len(t.split()) > max_words:
            continue
        k = t.lower()
        if k in seen:
            continue
        out.append(t); seen.add(k)
    return out

# Prompt builders
_SYSTEM_TEXT = (
    "You create short, concrete scenario sentences that exemplify a given concept "
    "in varied, realistic contexts. Output ONLY a numbered list, one per line, no extra text."
)

def _initial_prompt_text(label_for_prompt: str, n: int) -> str:
    return (
        f"{_SYSTEM_TEXT}\n\n"
        f'Concept: "{label_for_prompt}"\n'
        f"Write EXACTLY {n} distinct, concise scenario sentences (≤ 20 words each).\n"
        f"Format: 1. ..., 2. ..., …, {n}. ...\n"
        f"Do not start a new list after {n}. Do not add quotes or JSON.\n\n"
        f"1."
    )

def _continue_prompt_text(label_for_prompt: str, existing: list[str], n: int) -> str:
    k = len(existing)
    prefix = "\n".join(f"{i+1}. {existing[i]}" for i in range(k))
    return (
        f'Concept: "{label_for_prompt}"\n'
        f"Continue the numbered list with items {k+1} to {n}. "
        f"Output ONLY the numbered lines.\n\n"
        f"{prefix}\n{k+1}."
    )

#  batched generate
def _generate_texts_batch(prompt_texts: list[str], do_sample: bool, temperature: float,
                          top_p: float, max_new_tokens: int) -> list[str]:
    # Explicit max_length to silence truncation warnings and keep inputs sane
    inp = gen_tok(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,     
    ).to(next(gen_mdl.parameters()).device)

    input_len = inp.input_ids.shape[1]
    gen_mdl.eval()

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        eos_token_id=gen_tok.eos_token_id,
        use_cache=True,
        num_beams=1,
        do_sample=False,
        logits_processor=LogitsProcessorList([SafeLogitsProcessor()]),
    )
    if do_sample:
        
        temperature = float(max(1e-3, min(temperature, 5.0)))
        top_p = float(max(1e-6, min(top_p, 1.0)))
        gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p))

    
    try:
        with torch.inference_mode():
            out = gen_mdl.generate(**inp, **gen_kwargs)
    except Exception as e:
        if do_sample:
            # fallback: greedy decode for this batch
            gen_kwargs.update(dict(do_sample=False))
            with torch.inference_mode():
                out = gen_mdl.generate(**inp, **gen_kwargs)
        else:
            raise

    only_gen_ids = out[:, input_len:]
    texts = gen_tok.batch_decode(only_gen_ids, skip_special_tokens=True)
    return texts


# batched stimuli synth 
def llm_synthesize_stimuli_batch(cids: list[int], labels: list[str], n: int = 30,
                                 max_new_tokens: int | None = None,
                                 temperature_sampling: float = 0.7, top_p_sampling: float = 0.9,
                                 continue_tries: int = 2,
                                 gen_batch_size: int = 32) -> dict[int, list[str]]:
    assert len(cids) == len(labels)
    if max_new_tokens is None:
        max_new_tokens = min(12 * n + 16, 320)

    # Normalize labels for prompting
    norm_labels = [_normalize_label_for_prompt(lbl) for lbl in labels]

    # Storage
    results: dict[int, list[str]] = {cid: [] for cid in cids}

    #  PASS 1: greedy initial 
    for s in range(0, len(cids), gen_batch_size):
        idxs = list(range(s, min(s + gen_batch_size, len(cids))))
        prompts_text = [_initial_prompt_text(norm_labels[i], n) for i in idxs]
        texts = _generate_texts_batch(prompts_text, do_sample=False, temperature=0.0,
                                      top_p=1.0, max_new_tokens=max_new_tokens)
        for k, i in enumerate(idxs):
            parsed = parse_numbered_or_bulleted(texts[k], n)
            filtered = _post_filter_stimuli(labels[i], parsed)
            # merge unique into results
            seen = set(x.lower() for x in results[cids[i]])
            for sline in filtered:
                if sline.lower() not in seen:
                    results[cids[i]].append(sline); seen.add(sline.lower())
            # cap
            if len(results[cids[i]]) > n:
                results[cids[i]] = results[cids[i]][:n]

    # PASS 2: sampling retry for those still short 
    need_idxs = [i for i, cid in enumerate(cids) if len(results[cid]) < n]
    if need_idxs:
        for s in range(0, len(need_idxs), gen_batch_size):
            batch = need_idxs[s: s + gen_batch_size]
            prompts_text = [_initial_prompt_text(norm_labels[i], n) for i in batch]
            texts = _generate_texts_batch(prompts_text, do_sample=True,
                                          temperature=temperature_sampling, top_p=top_p_sampling,
                                          max_new_tokens=max_new_tokens)
            for k, i in enumerate(batch):
                parsed = parse_numbered_or_bulleted(texts[k], n)
                filtered = _post_filter_stimuli(labels[i], parsed)
                seen = set(x.lower() for x in results[cids[i]])
                for sline in filtered:
                    if sline.lower() not in seen:
                        results[cids[i]].append(sline); seen.add(sline.lower())
                    if len(results[cids[i]]) >= n:
                        break

    # PASS 3: continuation top-up (couple tries) 
    tries = 0
    while tries < continue_tries:
        need_idxs = [i for i, cid in enumerate(cids) if len(results[cid]) < n]
        if not need_idxs:
            break
        for s in range(0, len(need_idxs), gen_batch_size):
            batch = need_idxs[s: s + gen_batch_size]
            prompts_text = []
            caps = []
            for i in batch:
                remain = n - len(results[cids[i]])
                cap = min(12 * remain + 16, 200)
                caps.append(cap)
                prompts_text.append(_continue_prompt_text(norm_labels[i], results[cids[i]], n))
            texts = _generate_texts_batch(prompts_text, do_sample=False,
                                          temperature=0.0, top_p=1.0,
                                          max_new_tokens=max(caps) if caps else 200)
            for k, i in enumerate(batch):
                parsed = parse_numbered_or_bulleted(texts[k], n)
                filtered = _post_filter_stimuli(labels[i], parsed)
                seen = set(x.lower() for x in results[cids[i]])
                for sline in filtered:
                    if sline.lower() not in seen:
                        results[cids[i]].append(sline); seen.add(sline.lower())
                    if len(results[cids[i]]) >= n:
                        break
        tries += 1

    # trim to exactly n
    for cid in results:
        results[cid] = results[cid][:n]
    del norm_labels, prompts_text, texts, seen, filtered, parsed
    gc.collect(); torch.cuda.empty_cache()
    return results


# Hungarian uniqueness

def assign_unique_labels_hungarian(chunk_results, used_labels_global, top_k=10):
    from scipy.optimize import linear_sum_assignment
    all_labels = []
    label_to_idx = {}
    per_cid_cands = {}
    for cid, info in chunk_results.items():
        cands = [(l, s) for (l, s) in info.get("labels", []) if l not in used_labels_global]
        per_cid_cands[cid] = cands
        for l, _ in cands:
            if l not in label_to_idx:
                label_to_idx[l] = len(all_labels)
                all_labels.append(l)
    if not all_labels:
        return {cid: "unlabeled" for cid in chunk_results.keys()}
    cids = list(chunk_results.keys())
    C = np.ones((len(cids), len(all_labels)), dtype=np.float32)
    for i, cid in enumerate(cids):
        cands = per_cid_cands[cid]
        if not cands: continue
        max_s = max(s for _, s in cands) or 1e-6
        for l, s in cands:
            j = label_to_idx[l]
            norm = max(0.0, min(1.0, s / max_s))
            C[i, j] = 1.0 - norm
    row_idx, col_idx = linear_sum_assignment(C)
    assignment = {}
    unassigned = []
    for r, c in zip(row_idx, col_idx):
        cid = cids[r]
        if np.isclose(C[r, c], 1.0):
            unassigned.append(cid)
        else:
            lbl = all_labels[c]
            assignment[cid] = lbl
            used_labels_global.add(lbl)
    for cid in unassigned:
        cands = per_cid_cands.get(cid, [])
        assignment[cid] = cands[0][0] if cands else "unlabeled"
    return assignment

rng = np.random.default_rng(SEED)
def pick_cross_concept_negs_for_chunk(synth_by_cid, chunk_ids, k_concepts_each=40, m_prompts_per_concept=3):
    
    # Build negatives from synthesized stimuli (GLOBAL across ALL concepts).
    # Returns: dict[cid] -> list[str] (templated)
    
    all_cids = set(synth_by_cid.keys())
    neg_per_cid = {}
    for cid in chunk_ids:
        pool = list(all_cids - {cid})
        rng.shuffle(pool)
        take = pool[:k_concepts_each]
        negs = []
        for oc in take:
            plist = synth_by_cid.get(oc, [])
            if not plist: continue
            k = min(m_prompts_per_concept, len(plist))
            idxs = rng.choice(len(plist), size=k, replace=False)
            for i in idxs:
                negs.append(make_template(label=f"NEG-{oc}", stimulus=plist[i]))
        neg_per_cid[cid] = negs
    return neg_per_cid


# 4) Pass A — label every concept with Hungarian uniqueness

reranker = load_reranker()
chunk_size = 16
concept_id_to_label = {}
used_labels_global = set()

print("Labeling concepts with Hungarian uniqueness...")
for i in tqdm(range(0, len(concept_ids), chunk_size), desc="Label chunks"):
    chunk_ids = concept_ids[i : i + chunk_size]
    chunk = {cid: concept_to_prompts[cid] for cid in chunk_ids}
    chunk_results = label_concepts_with_dictionary(
        chunk, label_index, embedder, top_k=100, weighted=True, use_clustering=True, reranker=reranker
    )
    uniq = assign_unique_labels_hungarian(chunk_results, used_labels_global, top_k=100)
    concept_id_to_label.update(uniq)
    gc.collect(); torch.cuda.empty_cache()


# 5) Pass B — synthesize stimuli for EVERY (uniquely labeled) concept (BATCHED)

print("Synthesizing stimuli for each labeled concept (batched)...")
synth_by_cid = {}
LABEL_BATCH = 8  
for i in tqdm(range(0, len(concept_ids), LABEL_BATCH), desc="Stimuli (batched)"):
    batch_cids_all = concept_ids[i:i+LABEL_BATCH]
    batch_cids = [cid for cid in batch_cids_all if concept_id_to_label.get(cid, "unlabeled") != "unlabeled"]
    if not batch_cids:
        continue
    batch_labels = [concept_id_to_label[cid] for cid in batch_cids]
    batch_results = llm_synthesize_stimuli_batch(
        batch_cids, batch_labels, n=31,
        max_new_tokens=700,           
        temperature_sampling=0.85, top_p_sampling=0.9,
        continue_tries=2,
        gen_batch_size=8,              
    )

    synth_by_cid.update(batch_results)

# Fill any unlabeled with []
for cid in concept_ids:
    if cid not in synth_by_cid:
        synth_by_cid[cid] = []

# Free generator memory before encoding
del gen_tok, gen_mdl
gc.collect(); torch.cuda.empty_cache()


# 6) Pass C — compute per-layer concept directions (Option A)

OUTPUT_JSON = "concept_results.json"
with open(OUTPUT_JSON, "w") as f:
    f.write("[")

pairs_per_pos = 12               # diffs per positive prompt (12–24 is typical)
max_diffs_per_concept = 500      # cap for memory
batch_size = 16
max_length = 128

# probe to find layer count and choose PaCE window (last-29 .. last-11 inclusive)
ENCODER_NAME = "./llama2-7b-local"
tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
model = AutoModel.from_pretrained(
    ENCODER_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
    output_hidden_states=True,
)
model.gradient_checkpointing_enable()
tokenizer.pad_token = tokenizer.eos_token

_probe = tokenizer(["hello"], return_tensors="pt").to(device)
with torch.no_grad():
    _out = model(**_probe, output_hidden_states=True)
n_mods = len(_out.hidden_states)
start_layer = max(0, n_mods - 29)
end_layer = max(0, n_mods - 11)
layer_indices = list(range(start_layer, end_layer + 1))
del _probe, _out; torch.cuda.empty_cache()
print(f"Using layers [{layer_indices[0]}..{layer_indices[-1]}] (count={len(layer_indices)})")

for i in tqdm(range(0, len(concept_ids), chunk_size), desc="Direction chunks"):
    chunk_ids = concept_ids[i : i + chunk_size]

    # Positives (templated synthesized stimuli)
    pos_prompts_per_cid = {}
    for cid in chunk_ids:
        label = concept_id_to_label.get(cid, "unlabeled")
        templated = [make_template(label, s) for s in synth_by_cid.get(cid, [])]
        pos_prompts_per_cid[cid] = templated

    # Negatives (global cross-concept, also templated for symmetry)
    neg_texts_per_cid = pick_cross_concept_negs_for_chunk(
        synth_by_cid, chunk_ids, k_concepts_each=40, m_prompts_per_concept=3
    )

    # Flatten positives and negatives, keep ranges per cid
    pos_prompts_flat, pos_index_ranges = [], {}
    idx_ptr = 0
    for cid in chunk_ids:
        templ = pos_prompts_per_cid.get(cid, [])
        start_idx = idx_ptr
        pos_prompts_flat.extend(templ)
        idx_ptr += len(templ)
        pos_index_ranges[cid] = (start_idx, idx_ptr)

    neg_prompts_flat, neg_index_ranges = [], {}
    for cid in chunk_ids:
        start_idx = len(pos_prompts_flat) + len(neg_prompts_flat)
        neg_prompts_flat.extend(neg_texts_per_cid.get(cid, []))
        end_idx = len(pos_prompts_flat) + len(neg_prompts_flat)
        neg_index_ranges[cid] = (start_idx, end_idx)

    all_model_prompts = pos_prompts_flat + neg_prompts_flat
    if not all_model_prompts:
        continue

    # eencode
    layer_activations = compute_last_token_activations(
        all_model_prompts, model, tokenizer, device, layer_indices,
        batch_size=batch_size, max_length=max_length
    )
    n_layers_selected = len(layer_activations)

    # build results for this chunk
    chunk_results_out = []
    for cid in tqdm(chunk_ids, desc="Per-cid PCA", leave=False):
        label = concept_id_to_label.get(cid, "unlabeled")
        top_prompts = synth_by_cid.get(cid, [])
        sbert_emb = embedder.encode(label, convert_to_numpy=True).tolist() if label != "unlabeled" else None

        concept_vectors = {}
        if label != "unlabeled" and len(top_prompts) > 0:
            pos_start, pos_end = pos_index_ranges[cid]
            pos_indices = np.arange(pos_start, pos_end)

            neg_start, neg_end = neg_index_ranges[cid]
            cid_neg_indices = np.arange(neg_start, neg_end)

            for li, abs_layer_idx in enumerate(layer_indices):
                layer_arr = layer_activations[li]  # (N_all, d)
                diffs = []

                if pos_indices.size > 0 and cid_neg_indices.size > 0:
                    for p_idx in pos_indices:
                        nn = min(pairs_per_pos, cid_neg_indices.size)
                        picks = rng.choice(cid_neg_indices, size=nn, replace=False)
                        P = layer_arr[p_idx][None, :]                 # (1,d)
                        N = layer_arr[picks]                           # (nn,d)
                        D = P - N                                      # (nn,d)
                        D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
                        diffs.append(D)

                if diffs:
                    X = np.vstack(diffs)
                    if X.shape[0] > max_diffs_per_concept:
                        X = X[:max_diffs_per_concept]
                    try:
                        v = PCA(n_components=1, svd_solver="auto", random_state=SEED).fit(X).components_[0]
                    except Exception:
                        v = X.mean(0)
                    v = v / max(np.linalg.norm(v), 1e-12)
                else:
                    v = np.zeros(layer_arr.shape[1], dtype=np.float32)

                concept_vectors[f"layer_{abs_layer_idx}"] = v.astype(np.float32).tolist()

        chunk_results_out.append({
            "concept_id": cid,
            "label": label,                   # unique (Hungarian)
            "top_prompts": top_prompts,       # synthesized stimuli (cleaner then the noisy ones we had before)
            "sbert_embedding": sbert_emb,
            "llama_embeddings": concept_vectors
        })

        gc.collect(); torch.cuda.empty_cache()

    # append to file
    with open(OUTPUT_JSON, "a") as f:
        for j, result in enumerate(chunk_results_out):
            if i > 0 or j > 0:
                f.write(",")
            json.dump(result, f, indent=2)
            f.write("\n")

    gc.collect(); torch.cuda.empty_cache()
    del chunk_results_out, pos_prompts_per_cid, neg_texts_per_cid, pos_prompts_flat, neg_prompts_flat

with open(OUTPUT_JSON, "a") as f:
    f.write("]")

print(f"✅ Stage 1.2 done. Concept embeddings saved to {OUTPUT_JSON}")

del model, tokenizer, embedder, label_index
gc.collect(); torch.cuda.empty_cache()
