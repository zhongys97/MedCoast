import os
import json
import pickle
import random
import math
import argparse
import webdataset as wds
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


api_keys_json_path = os.path.join(parent_dir, 'api_keys.json')
with open(api_keys_json_path) as f:
    data = json.load(f)
    os.environ["OPENAI_API_KEY"] = data["OPENAI_API_KEY"]
    if "HF_HOME" in data.keys():
        os.environ["HF_HOME"] = os.path.expanduser(data["HF_HOME"])
    os.environ["HUGGINGFACE_HUB_TOKEN"] = data["HUGGINGFACE_HUB_TOKEN"]
    os.environ["GEMINI_API_KEY"] = data["GEMINI_API_KEY"]
    os.environ["ANTHROPIC_API_KEY"] = data["ANTHROPIC_API_KEY"]

from transformers import AutoTokenizer

# --- CONFIGURATION ---
TOKENIZER_ID = "answerdotai/ModernBERT-base" 
MAX_LENGTH = 2048  
TRUNCATION_SIDE = "right"

# -----------------------
# Logic from your original scripts
# -----------------------
def format_question(qa: dict) -> str:
    q = str(qa.get("question", "")).strip()
    if not qa.get("is_mcq", False):
        return q
    options = ""
    for opt, txt in qa.get("options", {}).items():
        options += f" {opt}. {txt.strip()}"
    return q + options

def build_prefix_from_path(tree, path_node_ids: List[str]) -> str:
    steps = []
    for nid in path_node_ids:
        nd = tree.get_node(nid)
        if nd and nd.data:
            rs = nd.data.get("rep_step")
            if rs: steps.append(str(rs).strip())
    return " <|step|> " + " <|step|> ".join(steps) if steps else ""

def smoothed_value(node_data: dict, weights=(0.5, 0.5)) -> float:
    n = node_data.get("n", 0)
    nc = node_data.get("n_correct", 0)
    pct = (float(nc) + 1.0) / (float(n) + 2.0) if n >= 0 else 0.5
    dr = node_data.get("depth_ratio_to_shortest_correct_leaf", 0.0) or 0.0
    return weights[0] * pct + weights[1] * dr

# -----------------------
# Modified Processing Worker
# -----------------------
def process_tree_to_samples(args):
    # Unpack args including the local path to the tokenizer
    tp, ds_name, qa_idx, qa_json, min_abs_gap, weights, local_tokenizer_path = args
    samples = []
    
    try:
        # Load tokenizer from local cache to avoid HTTP 429 Rate Limiting
        tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path, local_files_only=True)
        tokenizer.truncation_side = TRUNCATION_SIDE
        
        with open(tp, "rb") as f:
            tree = pickle.load(f)
        
        q_text = format_question(qa_json)
        
        for parent in tree.all_nodes_itr():
            pid = parent.identifier
            children = tree.children(pid)
            if len(children) < 2: continue

            path = [x for x in list(tree.rsearch(pid))[::-1] if x != "root"]
            parent_prefix = build_prefix_from_path(tree, path)

            scored = []
            for c in children:
                val = smoothed_value(tree.get_node(c.identifier).data, weights)
                scored.append((c.identifier, val))
            scored.sort(key=lambda x: x[1], reverse=True)

            for i in range(len(scored) - 1):
                ni, vi = scored[i]
                for j in range(i + 1, len(scored)):
                    nj, vj = scored[j]
                    gap = vi - vj
                    if gap < min_abs_gap: continue

                    sampling_prob = min(gap / 0.3, 0.5)
                    
                    if random.random() < sampling_prob:
                        pos_step = str((tree.get_node(ni).data or {}).get("rep_step", "")).strip()
                        neg_step = str((tree.get_node(nj).data or {}).get("rep_step", "")).strip()
                        
                        p_str = (parent_prefix + " <|step|> " + pos_step).strip() if pos_step else parent_prefix
                        n_str = (parent_prefix + " <|step|> " + neg_step).strip() if neg_step else parent_prefix

                        # --- Pairwise Tokenization (Pre-calculated for Efficiency) ---
                        # We store IDs as lists to offload CPU work from the training loop
                        enc_pos = tokenizer(
                            q_text, p_str,
                            truncation=True,
                            max_length=MAX_LENGTH,
                            add_special_tokens=True
                        )
                        enc_neg = tokenizer(
                            q_text, n_str,
                            truncation=True,
                            max_length=MAX_LENGTH,
                            add_special_tokens=True
                        )

                        samples.append({
                            "__key__": f"{ds_name}_{qa_idx}_{ni}_{nj}",
                            "json": {
                                "input_ids_pos": enc_pos["input_ids"],
                                "attention_mask_pos": enc_pos["attention_mask"],
                                "input_ids_neg": enc_neg["input_ids"],
                                "attention_mask_neg": enc_neg["attention_mask"],
                                "weight": float(gap)
                            }
                        })
        del tree
    except Exception as e:
        return None, str(e)
    
    random.shuffle(samples)

    return samples, None

# -----------------------
# Main Sharder
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, help="Dataset split (e.g. train, val, test)")
    parser.add_argument("--cutoff", type=float, required=True, help="Cutoff value used in tree directory name (e.g. 0.8)")
    parser.add_argument("--weight_correct", type=float, required=True, help="Weight for correct ratio in smoothed value (remainder goes to depth ratio)")
    parser.add_argument("--tree_base", default=None, help="Base directory for trees")
    parser.add_argument("--qa_base", default=None, help="Base directory for QA JSONs")
    parser.add_argument("--out_dir", default=None, help="Output directory for shards")
    args = parser.parse_args()

    split = args.split
    cutoff = args.cutoff
    weights = (args.weight_correct, 1.0 - args.weight_correct)
    tree_base = args.tree_base or f"{split}"
    qa_base = args.qa_base or f"./data/processed/{split}"
    out_dir = args.out_dir or f"./data/shards/{split}_cutoff{cutoff}_wc{weights[0]:.2f}"
    os.makedirs(out_dir, exist_ok=True)

    # 1. PRE-DOWNLOAD TOKENIZER (Prevents HTTP 429 Errors)
    print(f"Checking/Downloading tokenizer: {TOKENIZER_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    local_tokenizer_path = tokenizer.name_or_path 

    # 2. Load QA JSONs
    qa_data = {}
    for f in os.listdir(qa_base):
        if f.endswith('.json'):
            ds_name = f.split('_')[0]
            with open(os.path.join(qa_base, f)) as fin:
                qa_data[ds_name] = json.load(fin)

    # 3. Collect tree paths
    jobs = []
    for sub in os.listdir(tree_base):
        ds_path = os.path.join(tree_base, sub, f"cutoff_{cutoff}")
        if not os.path.isdir(ds_path): continue
        ds_name = sub.split('_')[0]
        for f in os.listdir(ds_path):
            if f.endswith('.pickle'):
                idx = int(f.split('_')[1])
                # Add the local tokenizer path to every job tuple
                jobs.append((
                    os.path.join(ds_path, f), 
                    ds_name, 
                    idx, 
                    qa_data[ds_name][idx], 
                    0.05,
                    weights,
                    local_tokenizer_path
                ))
    print(f"Total trees found: {len(jobs)}")

    random.shuffle(jobs)

    # 4. Process and Write Shards
    # Using 300MB maxsize to optimize filesystem I/O
    pattern = os.path.join(out_dir, "shard-%06d.tar")
    with wds.ShardWriter(pattern, maxsize=3e8, maxcount=100000) as sink:
        # Limit workers slightly to leave RAM for the OS/Main Process
        num_workers = max(1, (os.cpu_count() or 1) // 8)
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # USE executor.map instead of submit. 
            # chunksize=100 prevents the "hanging" by sending jobs in batches.
            # This returns an iterator that yields results in the order jobs were submitted.
            results_generator = executor.map(process_tree_to_samples, jobs, chunksize=100)
            
            # Wrap the generator in tqdm to see progress
            for samples, err in tqdm(results_generator, total=len(jobs), desc="Building Shards"):
                if samples:
                    for s in samples:
                        sink.write(s)
                elif err:
                    # Optional: log error to a file instead of flooding the screen
                    continue

if __name__ == "__main__":
    main()