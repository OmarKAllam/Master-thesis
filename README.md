# Concept Pipeline

Run the full pipeline (download → process → label → classify → PaCE) with a single command.

## Quick start

```bash
python 0.download_files.py && python 1.1_processing.py && python 1.2_labeling_v2.1.py && python 2_concept_classification.py && python 3_pace_v3.py
```

> Tip: Ensure your environment has the required packages installed (see `requirements.txt`). 
> If the pipeline uses gated models (e.g., Llama‑2), export your Hugging Face token first:
> `export HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX`

## Scripts order

1. `0.download_files.py` — downloads ConceptNet + LLaMA assets.
2. `1.1_processing.py` — data preprocessing & embeddings.
3. `1.2_labeling_v2.1.py` — label generation/assignment for concepts.
4. `2_concept_classification.py` — strict concept classification (Benign/Undesirable).
5. `3_pace_v3.py` — apply PaCE activation editing.
