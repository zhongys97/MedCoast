
import numpy as np
from typing import Any, Dict, List, Optional
from collections import defaultdict, deque
from treelib import Tree

def connected_components_by_threshold(cos: np.ndarray, threshold: float) -> List[List[int]]:
    n = cos.shape[0]
    if n == 0:
        return []
    visited = np.zeros(n, dtype=bool)
    clusters: List[List[int]] = []
    for start in range(n):
        if visited[start]:
            continue
        q = deque([start])
        visited[start] = True
        comp = [start]
        while q:
            u = q.popleft()
            neigh = np.where((cos[u] >= threshold) & (~visited))[0]
            for v in neigh.tolist():
                visited[v] = True
                q.append(v)
                comp.append(v)
        clusters.append(sorted(comp))
    return clusters



def print_all_paths_with_metrics(tree: Tree, root_id: str = "root") -> None:
    """
    For every node in the tree (1 line per node), print:
      - the rep_step path from root -> current node (excluding None root rep_step)
      - pct_children_leading_to_correct
      - depth_ratio_to_shortest_correct_leaf
    """
    for nid in tree.expand_tree(root_id, mode=Tree.WIDTH):
        node = tree.get_node(nid)

        # Build path: all rep_step along root->nid (skip None)
        path_ids = tree.rsearch(nid)  # yields nid -> ... -> root
        path_ids = list(reversed(list(path_ids)))  # root -> ... -> nid

        rep_steps = []
        for pid in path_ids:
            rs = tree.get_node(pid).data.get("rep_step", None)
            if rs is not None and rs != "<<MISSING_STEP>>":
                rep_steps.append(rs)

        pct = node.data.get("pct_children_leading_to_correct", None)
        ratio = node.data.get("depth_ratio_to_shortest_correct_leaf", None)

        # Print one line per node
        path_str = "  ->  ".join(rep_steps) if rep_steps else "(ROOT)"
        print(f"{path_str}\n    pct_children_leading_to_correct={pct} | depth_ratio_to_shortest_correct_leaf={ratio}\n")
