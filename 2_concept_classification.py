import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import json
import re
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from collections import Counter
import math

torch.cuda.empty_cache()

# Config
ALIGNMENT_TASK = "detoxification"
TASK_DESCRIPTIONS = {
    "detoxification": (
        "Task: Decide if the concept relates to toxic, harmful, violent, manipulative, or hateful content.\n"
        "UNDESIRABLE examples: harm, abuse, hate, maliciousness, crime, violence, manipulation, harassment, racism.\n"
        "BENIGN examples: neutral objects, neutral emotions, harmless activities, general facts."
    ),
    "faithfulness": (
        "Task: Decide if the concept tends to encourage fabrication or hallucination versus factual accuracy.\n"
        "UNDESIRABLE examples: fabrication, misinformation, hallucination, false claims.\n"
        "BENIGN examples: neutral factual content and grounded references."
    ),
    "sentiment": (
        "Task: Decide if the concept carries negative/hostile sentiment versus neutral/positive.\n"
        "UNDESIRABLE examples: anger, contempt, disgust, hostility, bigotry, aggression.\n"
        "BENIGN examples: neutral/positive sentiment such as friendship, gratitude, curiosity, respect."
    ),
}
TASK_DESC = TASK_DESCRIPTIONS.get(ALIGNMENT_TASK.lower(), TASK_DESCRIPTIONS["detoxification"])

# Load concepts
with open("concept_results.json") as f:
    concepts = json.load(f)

# Model setup
gpt_model_name = "./llama2-7b-local"  # or "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(gpt_model_name)
model = AutoModelForCausalLM.from_pretrained(
    gpt_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Few-shot
few_shot_examples = """
Examples:
Concept: "rifle"
Output: UNDESIRABLE

Concept: "ammunition"
Output: UNDESIRABLE

Concept: "browning_machine_gun"
Output: UNDESIRABLE

Concept: "armory"
Output: UNDESIRABLE

Concept: "kindness"
Output: BENIGN

Concept: "art_history"
Output: BENIGN

Concept: "self-defense"
Output: BENIGN

Concept: "knife"
Output: UNDESIRABLE

Concept: "hunting"
Output: UNDESIRABLE
"""

# Base templates
# BASE_TEMPLATES = [
#     "IGNORE PREVIOUS INSTRUCTIONS. You are a simple labeler.\n"
#     "Decision rule: {task_rule}\n"
#     "Write EXACTLY ONE WORD on the first non-empty line after 'Output:' — BENIGN or UNDESIRABLE — in UPPERCASE, and NOTHING ELSE.\n\n"
#     "{few_shot}\n"
#     "Contextual examples (from concept):\n{context}\n\nConcept: \"{LABEL}\"\nOutput:",
#     "PLEASE FOLLOW THE RULES STRICTLY. You are a simple labeler.\n"
#     "Decision rule: {task_rule}\n"
#     "Write EXACTLY ONE WORD on the first non-empty line after 'Output:' — BENIGN or UNDESIRABLE — in UPPERCASE, and NOTHING ELSE.\n\n"
#     "{few_shot}\n"
#     "Contextual examples (from concept):\n{context}\n\nConcept: \"{LABEL}\"\nOutput:",
#     "Short task: classify the single concept below.\n"
#     "Decision rule: {task_rule}\n"
#     "Write EXACTLY ONE WORD on the first non-empty line after 'Output:' — BENIGN or UNDESIRABLE — in UPPERCASE, and NOTHING ELSE.\n\n"
#     "{few_shot}\n"
#     "Contextual examples (from concept):\n{context}\n\nConcept: \"{LABEL}\"\nOutput:",
#     "STRICT FORMAT: ONE WORD ONLY (BENIGN or UNDESIRABLE).\n"
#     "Decision rule: {task_rule}\n"
#     "Write EXACTLY ONE WORD on the first non-empty line after 'Output:' — BENIGN or UNDESIRABLE — in UPPERCASE, and NOTHING ELSE.\n\n"
#     "{few_shot}\n"
#     "Contextual examples (from concept):\n{context}\n\nConcept: \"{LABEL}\"\nOutput:",
# ]
BASE_TEMPLATES = [
    
    "IGNORE PREVIOUS INSTRUCTIONS. You are a simple labeler.\n"
    "Decision rule: {task_rule}\n"
    "VALID OUTPUTS: BENIGN or UNDESIRABLE. Write EXACTLY ONE WORD after 'Output:' in UPPERCASE—nothing else.\n\n"
    "{few_shot}\n"
    "Contextual examples (from concept):\n{context}\n\nConcept: \"{LABEL}\"\nOutput:",

    
    "PLEASE FOLLOW THE RULES STRICTLY.\n"
    "Decision rule: {task_rule}\n"
    "Format: Output must be ONE WORD ONLY (BENIGN or UNDESIRABLE), all caps, no punctuation, no explanation.\n\n"
    "{few_shot}\n"
    "Context:\n{context}\n\nConcept: \"{LABEL}\"\nOutput:",

    
    "Short task: classify the single concept below.\n"
    "Decision rule: {task_rule}\n"
    "Respond with ONE TOKEN: BENIGN or UNDESIRABLE. NO EXTRA TEXT.\n\n"
    "{few_shot}\n"
    "Context:\n{context}\n\nConcept: \"{LABEL}\"\nOutput:",

    
    "CLASSIFICATION PROTOCOL\n"
    "Policy: {task_rule}\n"
    "Constraints:\n- Output exactly one label\n- Allowed labels: BENIGN, UNDESIRABLE\n- Uppercase only; no commentary\n\n"
    "{few_shot}\n"
    "Evidence excerpts:\n{context}\n\nConcept: \"{LABEL}\"\nOutput:",

    
    "Q: Given the policy below, classify the concept.\n"
    "Policy: {task_rule}\n"
    "A: Answer with exactly one uppercase word: BENIGN or UNDESIRABLE.\n\n"
    "{few_shot}\n"
    "Context snippets:\n{context}\n\nConcept: \"{LABEL}\"\nOutput:",

    
    "EXAMINER MODE: Grade the concept per the rule.\n"
    "Rule: {task_rule}\n"
    "Return exactly one of: BENIGN / UNDESIRABLE. Uppercase. Nothing else.\n\n"
    "{few_shot}\n"
    "Supporting context:\n{context}\n\nConcept: \"{LABEL}\"\nOutput:",

    
    "CHECKLIST:\n"
    "1) Apply decision rule: {task_rule}\n"
    "2) Choose ONLY ONE: BENIGN or UNDESIRABLE\n"
    "3) Output must be EXACTLY that word in uppercase\n\n"
    "{few_shot}\n"
    "Context samples:\n{context}\n\nConcept: \"{LABEL}\"\nOutput:",

    
    "INSTRUCTION: You will output ONLY a single uppercase label (BENIGN or UNDESIRABLE). Do NOT use JSON, quotes, or extra words.\n"
    "Decision rule: {task_rule}\n\n"
    "{few_shot}\n"
    "Context references:\n{context}\n\nConcept: \"{LABEL}\"\nOutput:",

    
    "TASK: Binary classification.\n"
    "Rule: {task_rule}\n"
    "Return exactly one uppercase label: BENIGN or UNDESIRABLE.\n\n"
    "{few_shot}\n"
    "Context:\n{context}\n\nConcept: \"{LABEL}\"\nOutput:",
]


# helpers
def format_top_prompts(tp, max_items=10, max_chars=2800):
    if not tp:
        return "- (no contextual examples provided)"
    if isinstance(tp, str):
        tp_list = [tp]
    else:
        tp_list = list(tp)
    tp_list = tp_list[:max_items]
    joined = "\n".join(f"- {p}" for p in tp_list)
    if len(joined) > max_chars:
        joined = joined[:max_chars] + "\n- [TRUNCATED]"
    return joined

def build_prompt(template_idx, label_text, context_text):
    templ = BASE_TEMPLATES[template_idx]
    return templ.format(task_rule=TASK_DESC, few_shot=few_shot_examples, context=context_text, LABEL=label_text)

def _tokenize_with_cache(text, *, _cache={}):
    # simple memoization to avoid re-tokenizing identical prompts
    if text in _cache:
        return _cache[text]
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    _cache[text] = enc
    return enc

def _score_continuations(input_ids, candidate_token_ids_lists):
    
    # Compute sum log-prob of each candidate continuation (list of token ids),
    # autoregressively, conditioning on the growing sequence.
    # Returns a list of floats (log-probs), one per candidate.
    
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    logps = []
    for cand in candidate_token_ids_lists:
        cur = input_ids
        total_lp = 0.0
        # roll forward one token at a time
        for t in cand:
            with torch.inference_mode():
                out = model(input_ids=cur)
                logits = out.logits[:, -1, :]  # next-token logits
                lp = torch.log_softmax(logits, dim=-1)[0, t].item()
            total_lp += lp
            next_tok = torch.tensor([[t]], device=device, dtype=cur.dtype)
            cur = torch.cat([cur, next_tok], dim=1)
        logps.append(total_lp)
        # free graph references
        del cur
    return logps

def classify_one_vote(label_text, top_prompts, template_idx, candidate_texts=(" BENIGN", " UNDESIRABLE")):
    
    # One 'vote' = build a prompt with a specific template, then score BENIGN vs UNDESIRABLE.
    # Returns: winner_label ("Benign"/"Undesirable"), per_option dict with logp & prob.
    
    context_text = format_top_prompts(top_prompts)
    prompt = build_prompt(template_idx, label_text, context_text)
    enc = _tokenize_with_cache(prompt)

    # Tokenize candidates (keep leading space!)
    cand_tok_lists = [tokenizer.encode(ct, add_special_tokens=False) for ct in candidate_texts]
    logps = _score_continuations(enc["input_ids"], cand_tok_lists)

    # Argmax decision (0=BENIGN, 1=UNDESIRABLE)
    idx = 0 if logps[0] >= logps[1] else 1
    label = "Benign" if idx == 0 else "Undesirable"

    # softmax over two options for nice reporting
    
    p0, p1 = math.exp(logps[0]), math.exp(logps[1])
    z = p0 + p1
    probs = [p0 / z, p1 / z]

    return label, {
        "BENIGN": {"logp": logps[0], "p": probs[0]},
        "UNDESIRABLE": {"logp": logps[1], "p": probs[1]},
        "template_idx": template_idx
    }

def classify_majority(label_text, top_prompts=None, template_indices=None, aggregate="majority"):
    
    # aggregate = "majority" => majority over per-template argmaxes.
    # aggregate = "logprob_sum" => sum log-probs over templates, argmax once.
    # Returns:
    #   final_label, votes_counter, details
    
    if template_indices is None:
        template_indices = list(range(len(BASE_TEMPLATES)))  # one vote per template

    votes = Counter()
    details = []  # keep per-vote logs
    sum_logps = {"BENIGN": 0.0, "UNDESIRABLE": 0.0}

    for tidx in template_indices:
        winner, info = classify_one_vote(label_text, top_prompts, tidx)
        votes[winner] += 1
        details.append(info)
        sum_logps["BENIGN"] += info["BENIGN"]["logp"]
        sum_logps["UNDESIRABLE"] += info["UNDESIRABLE"]["logp"]

    if aggregate == "majority":
        # tie-break by higher total log-prob if needed
        if votes["Benign"] > votes["Undesirable"]:
            final = "Benign"
        elif votes["Undesirable"] > votes["Benign"]:
            final = "Undesirable"
        else:
            final = "Benign" if sum_logps["BENIGN"] >= sum_logps["UNDESIRABLE"] else "Undesirable"
    elif aggregate == "logprob_sum":
        final = "Benign" if sum_logps["BENIGN"] >= sum_logps["UNDESIRABLE"] else "Undesirable"
    else:
        raise ValueError("aggregate must be 'majority' or 'logprob_sum'")

    # normalized probs from summed log-probs
    
    p0, p1 = math.exp(sum_logps["BENIGN"]), math.exp(sum_logps["UNDESIRABLE"])
    z = p0 + p1
    agg_probs = {"BENIGN": p0 / z, "UNDESIRABLE": p1 / z}

    return final, votes, {"per_vote": details, "sum_logps": sum_logps, "agg_probs": agg_probs, "aggregate": aggregate}

# Run over concepts
debug_examples = []
AGGREGATION = "majority"      # or "logprob_sum"
TEMPLATE_INDICES = list(range(len(BASE_TEMPLATES)))  # 9 votes (1 per template)
for c in tqdm(concepts, desc="Classifying"):
    lbl = c.get("label", None)
    if not lbl:
        c["category"] = "unlabeled"
        continue

    top_prompts = c.get("top_prompts", c.get("prompts", None))
    if isinstance(top_prompts, str):
        top_prompts = [top_prompts]

    final, votes, info = classify_majority(lbl, top_prompts=top_prompts,
                                           template_indices=TEMPLATE_INDICES,
                                           aggregate=AGGREGATION)
    c["category"] = final
    c["label_scores"] = info  # keep optional diagnostics
    print(f"{lbl}: {final}  (votes: {dict(votes)}  agg_probs={info['agg_probs']})")


with open("concept_results_classified.json", "w") as f:
    json.dump(concepts, f, indent=2)

print("✅ Done. Saved to concept_results_classified.json")
