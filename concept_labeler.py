from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm

import re
# Embedding / reranking
from sentence_transformers import SentenceTransformer
try:
    from sentence_transformers import CrossEncoder
    _HAS_CROSS = True
except Exception:
    _HAS_CROSS = False

# Index librariees
try:
    import faiss  # faiss-cpu
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize



def _l2(x: np.ndarray) -> np.ndarray:
    return normalize(x, norm="l2", axis=1)


def _safe_weights(similarities: List[float]) -> np.ndarray:
    w = np.asarray(similarities, dtype=np.float32)
    w = np.clip(w, 1e-6, None)  
    w = w / w.sum()
    return w


# Embedding models
def load_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:

    return SentenceTransformer(model_name,device="cuda")


def load_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Optional[CrossEncoder]:

    if not _HAS_CROSS:
        return None
    try:
        return CrossEncoder(model_name)
    except Exception:
        return None


# Dictionary / Index
class LabelIndex:
    
    # Wraps a FAISS or sklearn index over a labeled dictionary.
    
    def __init__(self, label_texts: List[str], embeddings: np.ndarray, use_faiss: bool = True):
        assert len(label_texts) == embeddings.shape[0], "texts vs embeddings mismatch"
        self.label_texts = label_texts
        self.embeddings = _l2(embeddings.astype(np.float32))
        self.dim = self.embeddings.shape[1]
        self.backend = "faiss" if (use_faiss and _HAS_FAISS) else "sklearn"

        if self.backend == "faiss":
            if len(self.embeddings) < 100:
                self.index = faiss.IndexFlatIP(self.dim)
                self.index.add(self.embeddings)
            else:
                quantizer = faiss.IndexFlatIP(self.dim)
                nlist = 100
                self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
                self.index.train(self.embeddings)
                self.index.add(self.embeddings)

        else:
            self.index = NearestNeighbors(n_neighbors=min(50, len(label_texts)), metric="cosine")
            self.index.fit(self.embeddings)


    def search(self, query_vecs: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        
        # query_vecs: (B, d) L2-normalized
        # returns: (indices, scores) where scores ~ cosine similarity
        
        query_vecs = _l2(query_vecs.astype(np.float32))
        if self.backend == "faiss":
            scores, idx = self.index.search(query_vecs, top_k)
            return idx, scores
        else:
            distances, idx = self.index.kneighbors(query_vecs, n_neighbors=top_k, return_distance=True)
            scores = 1.0 - distances  # cosine similarity
            return idx, scores


def build_label_index(
    dictionary_texts: List[str],
    embedder: SentenceTransformer,
    batch_size: int = 256,
    use_faiss: bool = True
) -> LabelIndex:
    
    # Embeds the dictionary labels, builds an index.
    # dictionary_texts: list of canonical labels/phrases (can include multiword terms).
    
    embs = []
    for i in tqdm(range(0, len(dictionary_texts), batch_size),desc="build_label_index embedding"):
        chunk = dictionary_texts[i : i + batch_size]
        embs.append(embedder.encode(chunk, convert_to_numpy=True, normalize_embeddings=True))
    embs = np.vstack(embs)
    return LabelIndex(dictionary_texts, embs, use_faiss=use_faiss)


# Concept centroiding & clustering
def concept_centroid(
    prompts_and_scores: List[Tuple[str, float]],
    embedder: SentenceTransformer,
    weighted: bool = True
) -> np.ndarray:
    """
    Compute (optionally weighted) centroid embedding for a concept from its top prompts.
    prompts_and_scores: [(prompt, similarity_to_concept), ...]
    """
    texts = [p for p, _ in prompts_and_scores]
    sims = [s for _, s in prompts_and_scores]

    
    X = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    
    if weighted:
        w = _safe_weights(sims)
        centroid = (w[:, None] * X).sum(axis=0)
    else:
        centroid = X.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
    return centroid.astype(np.float32)


def maybe_split_polysemantic(
    prompts_and_scores: List[Tuple[str, float]],
    embedder: SentenceTransformer,
    max_clusters: int = 3,
    min_cluster_size: int = 3
) -> List[List[Tuple[str, float]]]:
    
    # Try to detect/polysemantic concepts by clustering their example prompts.
    # Returns a list of clusters (each is a list of (prompt, score)).
    # If clustering is not helpful, returns a single cluster (the original list).
    
    if len(prompts_and_scores) < max(min_cluster_size, 5):
        return [prompts_and_scores]

    texts = [p for p, _ in prompts_and_scores]
    X = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    best_k, best_score = 1, -1.0
    best_labels = np.zeros(len(texts), dtype=int)
    for k in range(2, max_clusters + 1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(X)
        # Ensure reasonable cluster sizes
        sizes = np.bincount(labels)
        if (sizes < min_cluster_size).any():
            continue
        try:
            score = silhouette_score(X, labels)
        except Exception:
            score = -1.0
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    if best_k == 1 or best_score < 0.05:
        # Not worth splitting
        return [prompts_and_scores]

    clusters = []
    for k in range(best_k):
        cluster_items = [prompts_and_scores[i] for i in range(len(texts)) if best_labels[i] == k]
        clusters.append(cluster_items)
    return clusters



def _tokset(text: str) -> set[str]:
    # simple alpha tokenizer. mirrors your other normalization
    return set(re.findall(r"[a-z]+", text.lower()))

def lexical_score(examples_text: str, label: str) -> float:
    
    # Tiny BM25-ish prior: fraction of label tokens that appear in the examplee
    
    ex_toks = _tokset(examples_text)
    lbl_toks = _tokset(label)
    if not lbl_toks:
        return 0.0
    overlap = sum(1.0 for t in lbl_toks if t in ex_toks)
    return overlap / len(lbl_toks)

def label_concepts_with_dictionary(
    concept_to_prompts: Dict[int, List[Tuple[str, float]]],
    label_index: LabelIndex,
    embedder: SentenceTransformer,
    top_k: int = 5,
    weighted: bool = True,
    use_clustering: bool = True,
    reranker: Optional[CrossEncoder] = None,
    alpha: float = 0.85,  # cosine weight; (1-alpha) is lexical prior weight
) -> Dict[int, Dict]:
    results = {}
    for cid, examples in tqdm(concept_to_prompts.items(), desc="label_concepts_with_dictionary fn"):
        clusters = maybe_split_polysemantic(examples, embedder) if use_clustering else [examples]

        cid_results = {"labels": [], "centroid": None, "clusters": []}
        cluster_centroids, cluster_label_candidates = [], []

        for cl in clusters:
            centroid = concept_centroid(cl, embedder, weighted=weighted)
            idx, scores = label_index.search(centroid[None, :], top_k=top_k)
            idx, scores = idx[0], scores[0]

            # pair up
            candidates = [(label_index.label_texts[i], float(scores[j])) for j, i in enumerate(idx)]

            
            # if reranker is not None and candidates:
            #     # one big "examples" string per cluster (cap for speed)
            #     ex_text = " ".join(p for p, _ in cl)[:2000]
            #     pairs = [(ex_text, cand) for cand, _ in candidates]
            #     ce_scores = reranker.predict(pairs)  # higher is better
            #     order = np.argsort(-ce_scores)
            #     candidates = [candidates[i] for i in order]
            
            if reranker is not None:
                
                ex_snips = sorted([p for p,_ in cl], key=len)[:10]
                pairs = [(ex, cand) for cand,_ in candidates for ex in ex_snips]
                ce = reranker.predict(pairs).reshape(len(candidates), -1).mean(axis=1)
                # fuse with cosine (and lexical, if you kept it)
                fused = []
                for (cand,(lbl,base)), ce_sc in zip(enumerate(candidates), ce):
                    fused.append((lbl, 0.7*base + 0.3*float(ce_sc)))
                candidates = sorted(fused, key=lambda x: -x[1])[:top_k]


            
            ex_text = " ".join(p for p, _ in cl)[:2000]
            rescored = []
            for lbl, cos_sc in candidates:
                # ensure cosine is in [0,1] if backend returns inner product
                cos_sc = float(np.clip(cos_sc, 0.0, 1.0))
                lex = lexical_score(ex_text, lbl)
                score = alpha * cos_sc + (1.0 - alpha) * lex
                rescored.append((lbl, score))
            # fall back gracefully if no candidates
            candidates = sorted(rescored, key=lambda x: -x[1])[:top_k] if rescored else []

            cluster_centroids.append(centroid)
            cluster_label_candidates.append(candidates)
            cid_results["clusters"].append({
                "centroid": centroid,
                "candidates": candidates,
                "size": len(cl),
                "examples": cl
            })

        # merge candidates across clusters (keep best score per label)
        merged: Dict[str, float] = {}
        for cand_list in cluster_label_candidates:
            for lbl, sc in cand_list:
                if lbl not in merged or sc > merged[lbl]:
                    merged[lbl] = sc
        merged_sorted = sorted(merged.items(), key=lambda x: -x[1])[:top_k]

        cid_results["labels"] = merged_sorted
        cid_results["centroid"] = np.mean(np.stack(cluster_centroids, axis=0), axis=0)
        results[cid] = cid_results

    return results


if __name__ == "__main__":
    

    # toy example for testing purposes
    concept_to_prompts = {
        0: [("Nuclear power plants split atoms to create energy.", 0.82),
            ("Uranium fission releases heat used for electricity.", 0.76),
            ("Reactor cores need control rods to manage chain reactions.", 0.73),
            ("Fusion aims to replicate the sun's process on Earth.", 0.60),
            ("Radioactive waste requires long-term storage.", 0.64)],
        1: [("Paris is the capital of France.", 0.81),
            ("The Eiffel Tower is a landmark in Paris.", 0.78),
            ("The Louvre Museum houses the Mona Lisa.", 0.74),
            ("France uses the Euro currency.", 0.63),
            ("Notre-Dame cathedral is located in Paris.", 0.71)]
    }

    
    embedder = load_embedder("sentence-transformers/all-mpnet-base-v2")

    
    dictionary_texts = [
        "nuclear energy", "nuclear fission", "nuclear fusion", "radioactive waste",
        "electricity generation", "power plant", "Paris", "Eiffel Tower", "France",
        "French culture", "Louvre Museum", "European capitals", "tourist attraction"
    ]
    label_index = build_label_index(dictionary_texts, embedder, use_faiss=True)

    # Optional: Cross-encoder reranking for extra precision
    reranker = load_reranker()  # may return None if not available

    # Label the concepts
    results = label_concepts_with_dictionary(
        concept_to_prompts,
        label_index,
        embedder,
        top_k=5,
        weighted=True,
        use_clustering=True,
        reranker=reranker
    )

    # Print nicely
    for cid, info in results.items():
        print(f"\n=== Concept {cid} ===")
        print("Top Labels:")
        for lbl, sc in info["labels"]:
            print(f"  - {lbl}  (score ~ {sc:.3f})")
        if len(info["clusters"]) > 1:
            print(f"Polysemy detected: {len(info['clusters'])} clusters")
        for k, cl in enumerate(info["clusters"]):
            print(f"  Cluster {k} (n={cl['size']}):")
            for ex, s in cl["examples"]:
                print(f"    {s:.3f} - {ex}")

