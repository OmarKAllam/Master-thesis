# Concept Pipeline

This repository contains a pipeline for downloading resources, processing data, labeling concepts, classifying them, and applying PaCE editing.

## Steps to Run

The pipeline consists of the following scripts:

1. **0.download_files.py**  
   Downloads ConceptNet data and LLaMA-2 model files.

2. **1.1_processing.py**  
   Preprocesses the dataset and prepares embeddings.

3. **1.2_labeling_v2.1.py**  
   Generates and assigns labels to discovered concepts.

4. **2_concept_classification.py**  
   Performs strict classification of concepts into *Benign* and *Undesirable*.

5. **3_pace_v3.py**  
   Applies Parsimonious Concept Engineering (PaCE) for activation editing.

## Run the Full Pipeline

You can run the entire pipeline in sequence with:

```bash
python 0.download_files.py && \
python 1.1_processing.py && \
python 1.2_labeling_v2.1.py && \
python 2_concept_classification.py && \
python 3_pace_v3.py
Requirements
Python 3.9+

PyTorch with CUDA (recommended)

Hugging Face transformers, accelerate, and huggingface_hub

Other dependencies listed in requirements.txt

Notes
Make sure you have access to meta-llama/Llama-2-7b-chat-hf on Hugging Face.

Set your Hugging Face token via:

bash
Copy code
export HF_TOKEN=hf_your_token_here
ConceptNet file will be saved in the project root, and the model will be downloaded into ./llama2-7b-local.
