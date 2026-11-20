#!/usr/bin/env python3
import os
import re
import json
import pickle
import random
import math
import gc
from dataclasses import dataclass
from typing import List, Tuple, Optional
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import sys

def log(msg):
    print(f"[STATUS] {msg}", flush=True)


# -----------------------
# Data structures
# -----------------------
@dataclass
class PairItem:
    tp: str
    ds_name: str
    qa_idx: int
    parent_path: List[str]
    pos_node: str
    neg_node: str
    pos_score: float
    neg_score: float
    weight: float


# -----------------------
# Helpers
# -----------------------
def infer_ds_name(path: str) -> str:
    p = path.lower()
    if "bioasq" in p:
        return "bioasq"
    if "medmcqa" in p:
        return "medmcqa"
    if "medqa" in p:
        return "medqa"
    return "pubmedqa"


def discover_tree_pickles(list_dataset_trees_dir: List[str], skip_medmcqa: bool = False) -> List[Tuple[str, str, int]]:
    out = []
    for ds_dir in list_dataset_trees_dir:
        ds_name = infer_ds_name(ds_dir)
        if skip_medmcqa and ds_name == "medmcqa":
            continue
        for f in os.listdir(ds_dir):
            if f.endswith(".pickle"):
                # expects qa_<idx>_tree.pickle
                try:
                    qa_idx = int(f.split("_")[1])
                except Exception:
                    continue
                out.append((os.path.join(ds_dir, f), ds_name, qa_idx))
    return out


def load_processed_qa(processed_qa_json_dir: str):
    # kept for parity / sanity check that jsons exist
    m = {}
    for f in os.listdir(processed_qa_json_dir):
        if f.endswith(".json"):
            ds_name = f.split("_")[0]
            with open(os.path.join(processed_qa_json_dir, f), "r") as fin:
                m[ds_name] = json.load(fin)
    return m


def smoothed_value(node_data: dict, label_weights=(1.0, 0.0)) -> float:
    # Laplace-smoothed pct_leaf_correct + optional depth_ratio
    n = node_data.get("n", 0)
    nc = node_data.get("n_correct", 0)
    pct = (float(nc) + 1.0) / (float(n) + 2.0) if n and n >= 0 else 0.5

    depth_ratio = node_data.get("depth_ratio_to_shortest_correct_leaf", 0.0)
    if depth_ratio is None:
        depth_ratio = 0.0

    w0, w1 = float(label_weights[0]), float(label_weights[1])
    return w0 * pct + w1 * float(depth_ratio)


def infer_split_and_cutoff(list_dataset_trees_dir: List[str]) -> Tuple[str, str]:
    cfg0 = list_dataset_trees_dir[0]
    split_type = "train" if "train" in cfg0 else "val"
    m = re.search(r"cutoff_(0\.\d+)", cfg0)
    cutoff_str = m.group(1) if m else "unknown"
    return split_type, cutoff_str


def build_pairs_for_tree(
    tp: str,
    ds_name: str,
    qa_idx: int,
    *,
    min_abs_gap: float,
    label_weights=(1.0, 0.0),
    M: int = 4,
    p_adj: float = 0.70,
    tail_max_cap: int = 12,
    seed: int = 13,
) -> List[PairItem]:
    """
    Fast O(K*M) per parent sampling.
    """
    # deterministic per-file randomness
    rng = random.Random(seed)

    with open(tp, "rb") as f:
        tree = pickle.load(f)

    pairs: List[PairItem] = []

    for parent in tree.all_nodes_itr():
        pid = parent.identifier

        # store parent_path once (root->parent, excluding "root")
        parent_path = list(tree.rsearch(pid))
        parent_path.reverse()
        parent_path = [x for x in parent_path if x != "root"]

        children = tree.children(pid)
        if len(children) < 2:
            continue

        scored = []
        for c in children:
            cd = tree.get_node(c.identifier).data
            v = float(smoothed_value(cd, label_weights=label_weights))
            scored.append((c.identifier, v))

        if len(scored) < 2:
            continue

        scored.sort(key=lambda x: x[1], reverse=True)


        tau = 0.6      # smaller -> more focused on "hard" (small gap) pairs
        lam = 0.3      # higher -> more uniform tail kept (more long tail)

        local = []
        for i in range(len(scored) - 1):
            ni, vi = scored[i]
            for j in range(i + 1, len(scored)):
                nj, vj = scored[j]
                gap = vi - vj
                if gap < min_abs_gap:
                    continue

                # keep with probability p(gap): near-uniform floor (lam) + exponential emphasis on small gaps
                p = lam + (1 - lam) * math.exp(-gap / tau)
                if random.random() < p:
                    local.append((ni, nj, vi, vj, gap))

        if not local:
            continue

        # K = len(scored)
        # tail_max = min(K - 1, tail_max_cap)
        # if tail_max <= 0:
        #     continue

        # local = []
        # for i in range(K - 1):
        #     ni, vi = scored[i]
        #     chosen_js = set()

        #     for _ in range(M):
        #         if rng.random() < p_adj:
        #             d = 1
        #             while d < tail_max and rng.random() < 0.35:
        #                 d += 1
        #             j = min(i + d, K - 1)
        #         else:
        #             lo = i + 1
        #             hi = min(K - 1, i + tail_max)
        #             if lo > hi:
        #                 continue
        #             j = rng.randint(lo, hi)

        #         if j <= i or j in chosen_js:
        #             continue
        #         chosen_js.add(j)

        #         nj, vj = scored[j]
        #         gap = vi - vj
        #         if gap >= min_abs_gap:
        #             local.append((ni, nj, float(gap)))

        # if not local:
        #     continue

        for (pos_n, neg_n, vi, vj, gap) in local:
            pairs.append(
                PairItem(
                    tp=tp,
                    ds_name=ds_name,
                    qa_idx=qa_idx,
                    parent_path=parent_path,
                    pos_node=pos_n,
                    neg_node=neg_n,
                    pos_score=float(vi),
                    neg_score=float(vj),
                    weight=float(gap),
                )
            )

    # free memory
    del tree
    # gc.collect()
    return pairs


def _worker(args):
    # unpack to keep ProcessPool pickling simple
    (tp, ds_name, qa_idx, min_abs_gap, label_weights, M, p_adj, tail_max_cap, seed) = args
    try:
        return build_pairs_for_tree(
            tp=tp,
            ds_name=ds_name,
            qa_idx=qa_idx,
            min_abs_gap=min_abs_gap,
            label_weights=label_weights,
            M=M,
            p_adj=p_adj,
            tail_max_cap=tail_max_cap,
            seed=seed,
        )
    except Exception as e:
        # return empty on error; parent prints
        return ("__ERROR__", tp, str(e))


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    # ---- YOU EDIT THESE TWO ----
    tree_dir = "./data/trees"
    split = "val"

    list_dataset_trees_dir = [
        os.path.join(tree_dir, split, "pubmedqa_" + split, "cutoff_0.8"),
        os.path.join(tree_dir, split, "medqa_" + split, "cutoff_0.8"),
        os.path.join(tree_dir, split, "medmcqa_" + split, "cutoff_0.8"),
        os.path.join(tree_dir, split, "bioasq_" + split, "cutoff_0.8"),
    ]
    processed_qa_json_dir = f"./data/processed/{split}"

    # ---- CONFIG (constants) ----
    min_abs_gap = 0.05
    label_weights = (0.5, 0.5)
    skip_medmcqa = False

    M = 4
    p_adj = 0.70
    tail_max_cap = 12

    # multiprocessing
    NUM_WORKERS = max(1, (os.cpu_count() or 1) // 2)  # tune: //2 is often safer on shared nodes
    BASE_SEED = 13

    # output path
    split_type, cutoff_str = infer_split_and_cutoff(list_dataset_trees_dir)
    out_dir = "./data/pairs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"sibling_pairs_{split_type}_cutoff_{cutoff_str}_w_{label_weights[0]}.pickle")

    # sanity check qa jsons exist/are readable
    _ = load_processed_qa(processed_qa_json_dir)

    # discover trees
    list_pickle_info = discover_tree_pickles(list_dataset_trees_dir, skip_medmcqa=skip_medmcqa)
    print(f"Found {len(list_pickle_info)} tree pickles. Building pairs with {NUM_WORKERS} workers...")

    # prepare worker args
    jobs = []
    for idx, (tp, ds_name, qa_idx) in enumerate(list_pickle_info):
        jobs.append((
            tp, ds_name, qa_idx,
            min_abs_gap, label_weights, M, p_adj, tail_max_cap,
            BASE_SEED + idx,  # per-file seed
        ))

    all_pairs: List[PairItem] = []
    n_err = 0
    total_trees = len(jobs)
    processed_count = 0

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futs = [ex.submit(_worker, j) for j in jobs]

        # Use a simpler update loop
        for fut in as_completed(futs):
            processed_count += 1
            res = fut.result()

            if isinstance(res, tuple) and len(res) == 3 and res[0] == "__ERROR__":
                n_err += 1
                log(f"Error in tree {res[1]}: {res[2]}")
            else:
                all_pairs.extend(res)

            # PERIODIC FEEDBACK (Every 1000 trees or 5% of total)
            if processed_count % 1000 == 0 or processed_count == total_trees:
                log(f"Progress: {processed_count}/{total_trees} trees ({(processed_count/total_trees)*100:.1f}%) | Pairs so far: {len(all_pairs)}")

    random.shuffle(all_pairs)

    with open(out_path, "wb") as f:
        pickle.dump(all_pairs, f)

    print(f"Done. Built {len(all_pairs)} pairs (errors={n_err})")
    print(f"Saved: {out_path}")
