from sentence_transformers import SentenceTransformer
import torch
import gc
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from itertools import islice
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import re
import random
from numpy.linalg import norm


# ------------------- 1. Load LLaMA -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("./llama2-7b-local")
model = AutoModel.from_pretrained(
    "./llama2-7b-local",
    torch_dtype=torch.float16 if device=="cuda" else torch.float32,
    device_map="auto",
    output_hidden_states=True,
    # attn_implementation="eager"
)
model.gradient_checkpointing_enable()

# ------------------- 2. Dataset -------------------
wikitext_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
dataset = load_dataset("databricks/databricks-dolly-15k")

def extract_english_prompts(data, max_samples=1000000):
    prompts = []
    for item in tqdm(data, desc="Extracting prompts"):
        instruction = item["instruction"]
        prompts.append(instruction)
        if len(prompts) >= max_samples:
            break
    return prompts

english_prompts = extract_english_prompts(dataset["train"], max_samples=20000)
prompts = [x["text"] for x in wikitext_dataset.select(range(100000)) if len(x["text"].strip())>0]
prompts += english_prompts

c4_stream = load_dataset("allenai/c4", "en", split="train", streaming=True)
c4_subset = islice(c4_stream, 100000)
for x in c4_subset:
    if len(x["text"].strip()) > 0:
        prompts.append(x["text"])


def is_valid_prompt(t: str) -> bool:
    t = t.strip()
    if re.match(r"^=+ .* =+$", t): return False       # wiki headings
    if re.match(r"^(Category|File|Image|Template):", t): return False
    if re.match(r"^\*+", t): return False             # bullets
    if len(t) < 40: return False

    letters = sum(ch.isalpha() for ch in t)
    digits  = sum(ch.isdigit() for ch in t)
    if letters / max(1,len(t)) < 0.55: return False   # too symbol/number heavy
    if digits  / max(1,len(t)) > 0.25: return False   # year lists, tables, stats
    return True


def clean_prompt(t: str) -> str:
    t = t.replace("@-@", "-")                           # Wikitext hyphen artifact
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s([.,;:!?])", r"\1", t)               # fix spaces before punc
    return t.strip()

prompts = [p for p in prompts if is_valid_prompt(p)]


prompts = [clean_prompt(p) for p in prompts]
print(f"Total prompts collected: {len(prompts)}")

def _norm_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

_seen = set()
_unique = []
for p in prompts:
    k = _norm_text(p)
    if k not in _seen:
        _seen.add(k)
        _unique.append(p)
prompts = _unique
print(f"After exact dedupe: {len(prompts)}")
RUN_NEAR_DUP = True
if RUN_NEAR_DUP and len(prompts) > 0:
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        try:
            import faiss  # for speed
            _HAS_FAISS = True
        except Exception:
            _HAS_FAISS = False

        THRESH = 0.92          # consider as duplicate if cosine >= THRESH
        BATCH = 512            # embedding batch size

        emb_model = SentenceTransformer("all-MiniLM-L6-v2")
        kept_prompts = []
        kept_embs = None

        if _HAS_FAISS:
            index = None  

        for i in tqdm(range(0, len(prompts), BATCH),desc='near duplicates'):
            batch = prompts[i:i+BATCH]
            embs = emb_model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)

            for p, e in zip(batch, embs):
                if kept_embs is None:
                    kept_prompts.append(p)
                    kept_embs = e[None, :]
                    if _HAS_FAISS:
                        d = kept_embs.shape[1]
                        index = faiss.IndexFlatIP(d)
                        index.add(kept_embs.astype(np.float32))
                    continue

                if _HAS_FAISS:
                    D, I = index.search(e.astype(np.float32)[None, :], 1)
                    max_sim = float(D[0, 0])
                else:
                    sims = kept_embs @ e
                    max_sim = float(sims.max())

                if max_sim < THRESH:
                    kept_prompts.append(p)
                    if _HAS_FAISS:
                        index.add(e.astype(np.float32)[None, :])
                    else:
                        kept_embs = np.vstack([kept_embs, e]) if kept_embs is not None else e[None, :]

        prompts = kept_prompts
        print(f"After near-dup dedupe: {len(prompts)}")
    except Exception as e:
        print(f"[near-dup dedupe skipped: {e}]")

del _unique, _seen
gc.collect()
torch.cuda.empty_cache()
with open("prompts.json", "w") as f:
    json.dump(prompts, f)

# ------------------- 3. Get bottleneck activations -------------------
layer_index = -1
all_activations = []
tokenizer.pad_token = tokenizer.eos_token
batch_size = 8 


tokenizer.pad_token = tokenizer.eos_token


SELECT_LAYERS = None   # None = last 12 averaged
BATCH = 16 * 2
MAX_LEN = 256  

with torch.inference_mode():
    # last non-special, late layers
    for i in tqdm(range(0, len(prompts), BATCH), desc="activations (last non-special, late layers)"):
        batch = prompts[i:i+BATCH]
        # build masks 
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(device)
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states

        input_ids = enc["input_ids"]               # (B,T)
        attn_mask = enc["attention_mask"].bool()   # (B,T)
        B, T = input_ids.shape

        sp_ids = tokenizer.all_special_ids or []
        if sp_ids:
            sp = torch.tensor(sp_ids, device=input_ids.device)
            is_special = (input_ids[..., None] == sp).any(-1)             # (B,T)
        else:
            is_special = torch.zeros_like(input_ids, dtype=torch.bool)

        
        flat = input_ids.view(-1).tolist()
        pieces = tokenizer.convert_ids_to_tokens(flat)
        # strip the leading space marker used by SP models
        pieces = [p[1:] if p.startswith("▁") else p for p in pieces]
        pieces = np.array(pieces, dtype=object).reshape(B, T)

        # a token is "word-like" if it contains ANY Unicode letter
        def has_letter(s: str) -> bool:
            
            return any(ch.isalpha() for ch in s)

        has_alpha_np = np.vectorize(has_letter, otypes=[bool])(pieces)
        keep_np = attn_mask.cpu().numpy() & (~is_special.cpu().numpy()) & has_alpha_np
        keep = torch.from_numpy(keep_np).to(device)

        
        r = torch.flip(keep.long(), dims=[1])
        off = r.argmax(dim=1)
        has_keep = keep.any(dim=1)
        last_idx = (T - 1) - off

        
        K = 64
        tail = keep[:, -K:]
        has_tail = tail.any(dim=1)
        r_tail = torch.flip(tail.long(), dims=[1])
        off_tail = r_tail.argmax(dim=1)
        idx_tail = (T - 1) - off_tail

        last_nonpad = attn_mask.long().sum(dim=1) - 1
        last_idx = torch.where(has_keep, last_idx,
                    torch.where(has_tail, idx_tail, last_nonpad)).clamp(min=0)
        used_fallback = ~(has_keep | has_tail)     # True only if no usable token at all

        # get activations from late layers and continue as before 
        layers = hs[-12:]
        gathered = [h[torch.arange(B, device=h.device), last_idx, :] for h in layers]  # list of (B, H)
        # L2-normalize each layer vector per example
        gathered_norm = []
        for g in gathered:
            g = g.float()
            denom = g.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            gathered_norm.append(g / denom)
        # Average normalized layers, then ReLU to enforce nonnegativity for NMF
        acts = torch.stack(gathered_norm, dim=0).mean(dim=0).relu().float()
        all_activations.append(acts.cpu().numpy().astype(np.float32))
        del acts, gathered_norm, gathered, layers , enc, out, hs, input_ids, attn_mask
        gc.collect()
        torch.cuda.empty_cache()


print("before vstack")
A = np.vstack(all_activations)
print(f"Activation matrix shape: {A.shape}")
np.save("activations.npy", A)
del all_activations, A
gc.collect()
torch.cuda.empty_cache()

# ------------------- 4. NMF -------------------
print("before loading")
A = np.load("activations.npy", mmap_mode="r").astype(np.float32)

# --- column scaling (store ORIGINAL scale vector) ---
print("before p95")
# Robust per-feature scale using p95 (less sensitive to outliers than max)
p95 = np.percentile(A, 95, axis=0)               # shape (p,)
scale = p95.astype(np.float32)
scale[scale <= 0] = 1.0                         

A_scaled = (A / scale[None, :]).astype(np.float32)

# keep these to map C back to original units later
col_scale_orig = scale.copy()


# --- prune low-variance columns 
print("computing column variance")
col_var = A_scaled.var(axis=0)                    # shape (p,)
thr = np.percentile(col_var, 15)                  # drop lowest-variance 15%
keep_cols = col_var > thr
kept = int(keep_cols.sum())
total = int(keep_cols.size)
print(f"keeping {kept}/{total} cols ({100*kept/total:.1f}%)")

A_red = A_scaled[:, keep_cols]
np.save("activations.reduced.npy", A_red)

# --- NMF ---
print("A_red shape is : ",A_red.shape)
k = 1024
nmf = NMF(
    n_components=min(k, A_red.shape[1], A_red.shape[0]),
    solver="mu",
    init="nndsvdar",
    max_iter=600,
    # max_iter=10,
    # alpha_W=1e-2, alpha_H=1e-2, l1_ratio=0.5,
    alpha_W=5e-2, alpha_H=0, 
    
    # tol=1e-4,
    verbose=True,
    random_state=42,
)
print("before fitting U")
U = nmf.fit_transform(A_red)                       # W (N, k)
print("before getting C_red")
C_red = nmf.components_                            # H (k, d_reduced)

# --- expand C back to scaled full dim, then unscale to ORIGINAL units ---
p = keep_cols.size
C_scaled = np.zeros((C_red.shape[0], p), dtype=C_red.dtype)
C_scaled[:, keep_cols] = C_red

print(f"Scaled concept matrix shape: {C_scaled.shape}")
C = C_scaled * col_scale_orig[None, :]               # unscale columns

print(f"Concept matrix shape: {C.shape}")
np.save("U.npy", U)
np.save("C.npy", C)


del U, C, A, C_scaled, col_scale_orig, A_scaled
gc.collect()
torch.cuda.empty_cache()

# ------------------- 5. Top-K prompts per concept -------------------
U = np.load("U.npy", mmap_mode="r").astype(np.float64)
C = np.load("C.npy", mmap_mode="r").astype(np.float64)
A = np.load("activations.npy", mmap_mode="r").astype(np.float64)

print("Did it converge early?", nmf.n_iter_ < nmf.max_iter)
print("Number of iterations used:", nmf.n_iter_)



reconstruction_error = norm(A - U @ C, 'fro')
print("Final reconstruction error:", reconstruction_error)

print("Stored in nmf.reconstruction_err_:", nmf.reconstruction_err_)

baseline = norm(A, 'fro')
rel_error = reconstruction_error / baseline
print("Relative reconstruction error:", rel_error)

rel_err_kept = norm(A[:, keep_cols] - U @ C_red, 'fro') / norm(A[:, keep_cols], 'fro')
print("Relative reconstruction error (kept cols):", rel_err_kept)


print("NMF used", nmf.n_iter_, "iterations.")
with open("prompts.json", "r") as f:
    prompts = json.load(f)





def _cosine_sims(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    
    # Compute cosine similarities between concept rows in C (k x d) and
    # prompt rows in A (N x d). Returns sims of shape (k, N).
    
    A = A.astype(np.float32, copy=False)
    C = C.astype(np.float32, copy=False)

    A_norm = np.linalg.norm(A, axis=1, keepdims=True)
    C_norm = np.linalg.norm(C, axis=1, keepdims=True)
    A_norm[A_norm == 0] = 1.0
    C_norm[C_norm == 0] = 1.0

    A_normalized = A / A_norm
    C_normalized = C / C_norm
    sims = C_normalized @ A_normalized.T  
    return sims


def topk_by_usage_with_cosine_scores(
    U: np.ndarray,
    prompts: list,
    *,
    A: np.ndarray = None,
    C: np.ndarray = None,
    sims: np.ndarray = None,
    k: int = 50,
    min_usage: float = 1e-6,
    score_transform: str | None = None,
    return_prompt_to_concepts: bool = False,
):
    
    # Select prompts by usage (top-K per concept via U[:, j]) but store the score
    # for each (concept, prompt) pair as the cosine similarity between C[j] and A[i].

    # Args:
    #     U : (N, k) usage matrix from NMF (rows: prompts, cols: concepts)
    #     prompts : length N
    #     A : (N, d) prompt activations (same feature space as C).
    #     C : (k, d) concept matrix
    #     sims : (k, N) optional precomputed cosine similarity matrix

    #     k : top-K prompts by usage to keep per concept
    #     min_usage : ignore prompts with usage <= this threshold
    #     score_transform : optional transform applied to cosine score per pair:
    #         - None  -> keep raw cosine in [-1, 1]
    #         - "shift01" -> map to [0, 1] via (s+1)/2
    #         - "clip0"   -> clip negatives to 0.0 (max(s, 0))
    #     return_prompt_to_concepts : also return inverse mapping if True

    
    N, k_cols = U.shape
    assert len(prompts) == N, "len(prompts) must equal U.shape[0]"

    if sims is None:
        assert A is not None and C is not None, "Provide either sims or both A and C."
        sims = _cosine_sims(A, C)  # (k, N)

    # choose scorer
    if score_transform is None:
        def _score(s): return float(s)
    elif score_transform == "shift01":
        def _score(s): return float((s + 1.0) * 0.5)
    elif score_transform == "clip0":
        def _score(s): return float(max(0.0, s))
    else:
        raise ValueError(f"Unknown score_transform: {score_transform}")

    concept_to_prompts = {}
    prompt_to_concepts = {} if return_prompt_to_concepts else None

    for j in range(k_cols):
        col = U[:, j]
        idx = np.where(col > min_usage)[0]
        if idx.size == 0:
            concept_to_prompts[j] = []
            continue

        # pick by usage
        order = idx[np.argsort(-col[idx])]
        take = order[: min(k, order.size)]

        # store cosine scores for the taken prompts
        scores = sims[j, take]  # shape (len(take),)
        pairs = []
        for pid, s in zip(take, scores):
            sc = _score(float(s))
            pairs.append((prompts[pid], sc))
            if return_prompt_to_concepts:
                prompt_to_concepts.setdefault(pid, []).append((j, sc))

        concept_to_prompts[j] = pairs

    if return_prompt_to_concepts:
        return concept_to_prompts, prompt_to_concepts
    return concept_to_prompts



# precompute sims:
sims = _cosine_sims(A, C)

# Then select by usage but attach cosine scores:
concept_to_prompts = topk_by_usage_with_cosine_scores(
    U, prompts, sims=sims, k=50, min_usage=1e-6, score_transform="shift01"
)

print("number of concepts: ", len(concept_to_prompts) )
with open("concept_to_prompts.json", "w") as f:
    json.dump(concept_to_prompts, f)



# diagnostics section
# After running topk_similarities
c2p = concept_to_prompts



# Pick 10 random concept IDs
sampled_cids = random.sample(list(c2p.keys()), 10)

for cid in sampled_cids:
    print("Top prompts for concept", cid," with number of prompts: " ,len(c2p[cid]))
    for p, s in c2p[cid][:10]:
        print("-", p[:200], "...", s)


embedder = SentenceTransformer("all-MiniLM-L6-v2")
def coherence_for_cid(cid, top_k=20):
    plist = c2p.get(cid, [])
    top = [p for p,_ in plist[:top_k]]
    if len(top) < 2:
        return 0.0
    E = embedder.encode(top, convert_to_numpy=True, normalize_embeddings=True)
    S = cosine_similarity(E)
    n = len(top)
    return float((S.sum() - n) / (n*(n-1)))

# Precompute sims for margin
A = np.load("activations.npy", mmap_mode="r").astype(np.float32)
A_norm = A / np.maximum(np.linalg.norm(A,axis=1,keepdims=True),1e-12)
C_norm = C / np.maximum(np.linalg.norm(C,axis=1,keepdims=True),1e-12)
S = C_norm @ A_norm.T  # (k, N)

rng = np.random.default_rng(0)
def top_margin(j, k=50, R=2000):
    sims = S[j]
    topk = np.sort(sims)[-k:]
    rand = sims[rng.choice(len(sims), size=min(R, len(sims)), replace=False)]
    return float(topk.mean() - rand.mean())

cohs = np.array([coherence_for_cid(j) for j in range(C.shape[0])])
margins = np.array([top_margin(j) for j in range(C.shape[0])])

# Usage strength
u_max = U.max(axis=0)                 # (k,)
thr_umax = np.percentile(u_max, 15)   # drop faintest 15%
usage_keep = u_max >= thr_umax

# Redundancy: pairwise concept cosine
G = cosine_similarity(C)              # (k,k)
np.fill_diagonal(G, 0.0)

# For any pair cos>0.9, keep the better one by (coh, margin, u_max)
dup_threshold = 0.9
order = np.argsort(-cohs)  # start from best coherence to make greedy selection
keep_mask = np.zeros(C.shape[0], dtype=bool)

for j in order:
    if keep_mask[j]:
        continue
    # if already banned by usage or metrics, skip early, we'll collect via 'final_keep'
    keep_mask[j] = True
    # ban its near duplicates
    dups = np.where(G[j] > dup_threshold)[0]
    keep_mask[dups] =  False  # ensure off

# Combine criteria
# Primary: coherence & margin, Secondary: usage, Then apply dedupe mask
metric_keep = (cohs >= 0.15) & (margins >= 0.01)
final_keep = metric_keep & usage_keep & keep_mask

print(f"Keeping {final_keep.sum()} / {C.shape[0]} atoms")

#  Apply and save
C_pruned = C[final_keep]
U_pruned = U[:, final_keep]
np.save("C.pruned.npy", C_pruned)
np.save("U.pruned.npy", U_pruned)

#  Rebuild concept_to_prompts.json with pruned U (indices are now 0..k'-1)
def topk_by_usage_pruned(Up, k=50, min_usage=1e-8):
    c2p = {}
    for j in range(Up.shape[1]):
        col = Up[:, j]
        idx = np.where(col > min_usage)[0]
        take = idx[np.argsort(-col[idx])][:k]
        c2p[j] = [(prompts[i], float(col[i])) for i in take]
    return c2p

c2p_pruned = topk_by_usage_pruned(U_pruned, k=50)
# with open("concept_to_prompts.json", "w") as f:
#     json.dump(c2p_pruned, f)

#  Report new medians to sanity-check
print("Median coherence after prune:", np.median([coherence_for_cid(j) for j in range(C_pruned.shape[0])]))
print("Median margin after prune:", np.median([top_margin(j) for j in range(C_pruned.shape[0])]))

del model, tokenizer, A, U, C, concept_to_prompts
gc.collect()
torch.cuda.empty_cache()

print("✅ Stage 1.1 done. Intermediate results saved to prompts.json, activations.npy, U.npy, C.npy, and concept_to_prompts.json")