import json
from common.utils import process_single_pred_response, grade_responses
from preprocessing.utils import connected_components_by_threshold
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import os
import json
from collections import defaultdict, deque
import uuid
import numpy as np
import pickle
from treelib import Tree


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


# MCQ medqa_0 val
# FRQ bioasq_0 val


class TrajectoriesToTreeBuilder:
    def __init__(self,
                dataset_name: str,
                split_name: str,
                processed_path: str = "./data/processed/",
                samples_path: str = "./data/samples/",
                chains_path: str = "./data/chains/",
                trees_path: str = "./data/trees/",
                sentence_transformer_model_name: str = None,
                device: str = None):

        """
        Initialize the TrajectoriesToTreeBuilder.

        Args:
        sentence_transformer_model_name: Name of the sentence transformer model for embeddings.
        device: Device to run grading model on for BERTscore.

        """

        if dataset_name not in ["BioASQ", "MedQA", "MedMCQA", "PubMedQA", "MedXpertQA", "MedQuAD", "MMLU"]:
            raise ValueError(f"Unknown dataset: {dataset_name}")


        self.dataset_name = dataset_name.lower()

        if split_name not in ["train", "val"]:
            raise ValueError(f"only training and validation splits are supported. Got: {split_name}")
        
        self.split_name = split_name.lower()

        if os.path.exists(processed_path) is False:
            raise ValueError(f"processed_path {processed_path} does not exist.")
        self.processed_path = processed_path

        if os.path.exists(samples_path) is False:
            raise ValueError(f"samples_path {samples_path} does not exist.")
        self.samples_path = samples_path

        self.chains_path = chains_path
        os.makedirs(self.chains_path, exist_ok=True)

        self.trees_path = trees_path
        os.makedirs(self.trees_path, exist_ok=True)


        qa_pairs_path = os.path.join(self.processed_path, self.split_name, f"{self.dataset_name}_{self.split_name}.json")
        with open(qa_pairs_path, 'r') as f:
            self.qa_pairs = json.load(f)

        
        self.sentence_transformer_model_name = sentence_transformer_model_name
        if sentence_transformer_model_name is not None:
            api_keys_json_path = os.path.join(parent_dir, 'api_keys.json')
            with open(api_keys_json_path) as f:
                data = json.load(f)
                os.environ["OPENAI_API_KEY"] = data["OPENAI_API_KEY"]
                if "HF_HOME" in data.keys():
                    os.environ["HF_HOME"] = os.path.expanduser(data["HF_HOME"])
                    
            from sentence_transformers import SentenceTransformer, util

            self.model = SentenceTransformer(sentence_transformer_model_name, device=device)
        
        self.device = device


    def traj_to_chains(self,
                       traj_file_paths: List[str],
                       ground_truth_qa_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        
        """
        
        Convert trajectories from files into chains with grading.
        Args:
        traj_file_paths: List of file paths containing trajectory JSONL data.
        ground_truth_qa_dict: Ground truth QA pair dictionary.
        """

        trajectories = []
        for path in traj_file_paths:
            with open(path, 'r') as f:
                trajectories.extend([json.loads(line) for line in f])

        ground_truth_qa = ground_truth_qa_dict

        list_of_chains = []

        gt_answer = ground_truth_qa["answer"]
        is_mcq = ground_truth_qa["is_mcq"]

        pred_answers_cur_qa = []
        gt_answers = []
        steps = []
        for traj in trajectories:
            try:
                out = process_single_pred_response(traj)
                if out["pred_answer"] == "":
                    continue
                pred_answers_cur_qa.append(out["pred_answer"])
                gt_answers.append(gt_answer)
                steps.append(out["steps"])
            except Exception as e:
                print(f"Error processing trajectory: {e}")
                continue

        try:
            gradings = grade_responses(is_mcq, pred_answers_cur_qa, gt_answers, device=self.device)

            for i in tqdm(range(len(pred_answers_cur_qa))):

                if is_mcq:
                    chain_dict = {
                        "steps": steps[i],
                        "pred_answer": pred_answers_cur_qa[i],
                        "gt_answer": gt_answer,
                        "grading": gradings[i]["correct"],
                    }
                else:
                    chain_dict = {
                        "steps": steps[i],
                        "pred_answer": pred_answers_cur_qa[i],
                        "gt_answer": gt_answer,
                        "grading": gradings[i]["rouge"]["rougeL"]["f1"] > 0.15, # based on stat, mean is 0.145
                    }
                list_of_chains.append(chain_dict)
        except Exception as e:
            print(f"Error during grading: {e}")
            return []

        return list_of_chains
    

    def _build_prelim_tree_from_chains(self,
                                chains: List[Dict[str, Any]],
                                *,
                                steps_key: str = "steps",
                                correct_key: str = "grading",
                                cutoff: float = 0.78,
                                max_depth: Optional[int] = None) -> Tree:
        
        from sentence_transformers import util

        def summarize(ids: List[int]) -> Dict[str, int]:
            n_correct = sum(1 for i in ids if bool(chains[i].get(correct_key, False)))
            return {"n": len(ids), "n_correct": n_correct, "n_incorrect": len(ids) - n_correct}

        tree = Tree()
        root_id = "root"
        all_ids = list(range(len(chains)))
        tree.create_node(
            tag="ROOT",
            identifier=root_id,
            data={"depth": -1, "rep_step": None, "answer_ids": all_ids, **summarize(all_ids)},
        )

        def recurse(parent_id: str, ids: List[int], depth: int) -> None:
            if max_depth is not None and depth > max_depth:
                return

            text_to_ids: Dict[str, List[int]] = defaultdict(list)
            for aid in ids:
                steps = chains[aid].get(steps_key, [])
                if depth < len(steps):
                    t = str(steps[depth]).strip()
                    if t:
                        text_to_ids[t].append(aid)

            if not text_to_ids:
                return

            uniq_texts = list(text_to_ids.keys())
            emb = self.model.encode(uniq_texts, convert_to_tensor=True, normalize_embeddings=True, batch_size=64)
            cos = util.cos_sim(emb, emb).cpu().numpy()
            comps = connected_components_by_threshold(cos, cutoff)

            for comp in comps:
                comp_texts = [uniq_texts[i] for i in comp]
                comp_ids: List[int] = []
                for t in comp_texts:
                    comp_ids.extend(text_to_ids[t])

                rep_step = max(comp_texts, key=lambda t: len(text_to_ids[t]))
                node_id = uuid.uuid4().hex
                stats = summarize(comp_ids)

                tree.create_node(
                    tag=f"[d{depth}] n={stats['n']} acc={stats['n_correct']}/{stats['n']}",
                    identifier=node_id,
                    parent=parent_id,
                    data={
                        "depth": depth,
                        "rep_step": rep_step,
                        "answer_ids": sorted(comp_ids),
                        **stats,
                    },
                )
                recurse(node_id, sorted(comp_ids), depth + 1)

        recurse(root_id, all_ids, 0)
        return tree

    @ staticmethod
    def _annotate_metrics(tree: Tree, root_id: str = "root") -> None:
        """
        Adds to node.data:
        - pct_children_leading_to_correct
        - depth_ratio_to_shortest_correct_leaf
        "Correct leaf" = leaf node with n_correct > 0.
        """
        def postorder(nid: str) -> None:
            node = tree.get_node(nid)
            children = tree.children(nid)
            for c in children:
                postorder(c.identifier)

            if not children:
                is_correct_leaf = node.data.get("n_correct", 0) > 0
                node.data["has_correct_leaf_subtree"] = bool(is_correct_leaf)
                node.data["min_correct_leaf_depth"] = node.data["depth"] if is_correct_leaf else None
                node.data["pct_children_leading_to_correct"] = None
            else:
                child_has = []
                child_min_depths = []
                for c in children:
                    cd = tree.get_node(c.identifier).data
                    child_has.append(bool(cd.get("has_correct_leaf_subtree", False)))
                    d = cd.get("min_correct_leaf_depth", None)
                    if d is not None:
                        child_min_depths.append(int(d))
                node.data["has_correct_leaf_subtree"] = any(child_has)
                node.data["min_correct_leaf_depth"] = min(child_min_depths) if child_min_depths else None
                node.data["pct_children_leading_to_correct"] = sum(child_has) / len(child_has)

            depth = node.data.get("depth", None)
            m = node.data.get("min_correct_leaf_depth", None)
            node.data["depth_ratio_to_shortest_correct_leaf"] = (
                None if (depth is None or depth < 0 or m is None) else (depth / m if m > 0 else 1.0)
            )

        postorder(root_id)

    

    def save_chains_to_jsonl(self) -> None:

        samples_dir_cur_split_cur_ds = os.path.join(self.samples_path, self.split_name, self.dataset_name + "_" + self.split_name)

        model_names = [f for f in os.listdir(samples_dir_cur_split_cur_ds) if os.path.isdir(os.path.join(samples_dir_cur_split_cur_ds, f))]
        
        import random
        random.shuffle(self.qa_pairs)

        for qa in self.qa_pairs:
            try:
                qa_id = str(qa["idx"])

                chains_output_path = os.path.join(self.chains_path, self.split_name, self.dataset_name + "_" + self.split_name)
                os.makedirs(chains_output_path, exist_ok=True)
                chains_file_path = os.path.join(chains_output_path, f"qa_{qa_id}_chains.jsonl")
                if os.path.exists(chains_file_path):
                    print(f"Chains file {chains_file_path} already exists. Skipping.")
                    continue
                traj_file_paths = []
                for model_name in model_names:
                
                    traj_file_path = os.path.join(self.samples_path, self.split_name, self.dataset_name + "_" + self.split_name, model_name, f"qa_{qa_id}_trajectories.jsonl")
                    if not os.path.exists(traj_file_path):
                        print(f"Trajectory file {traj_file_path} does not exist. Skipping.")
                        continue
                    traj_file_paths.append(traj_file_path)

                chains = self.traj_to_chains(traj_file_paths=traj_file_paths,
                                            ground_truth_qa_dict=qa)
            
                with open(chains_file_path, 'w') as f:
                    for chain in chains:
                        f.write(json.dumps(chain) + "\n")
                
            except Exception as e:
                print(e)
                continue
            

        
    def build_chains_and_trees(self,
                               cutoff: float,
                               build_chains: bool=True,
                               ) -> None:
        
        if build_chains:
            self.save_chains_to_jsonl()
        
        # read from chains files and build trees
        import random
        random.shuffle(self.qa_pairs)

        for qa in tqdm(self.qa_pairs):
            qa_id = str(qa["idx"])

            try:
                trees_output_path = os.path.join(self.trees_path, self.split_name, self.dataset_name + "_" + self.split_name, "cutoff_" + str(cutoff))
                os.makedirs(trees_output_path, exist_ok=True)
                tree_file_path = os.path.join(trees_output_path, f"qa_{qa_id}_tree.pickle")
                if os.path.exists(tree_file_path):
                    print(f"Tree file {tree_file_path} already exists. Skipping.")
                    continue

                chains_output_path = os.path.join(self.chains_path, self.split_name, self.dataset_name + "_" + self.split_name)
                chains_file_path = os.path.join(chains_output_path, f"qa_{qa_id}_chains.jsonl")
                if not os.path.exists(chains_file_path):
                    print(f"Chains file {chains_file_path} does not exist. Skipping.")
                    continue

                with open(chains_file_path, 'r') as f:
                    chains = [json.loads(line) for line in f]
                tree = self._build_prelim_tree_from_chains(chains, cutoff=cutoff, max_depth=20)
                self._annotate_metrics(tree)

                # Save tree as pickle
                with open(tree_file_path, 'wb') as f:
                    pickle.dump(tree, f)
            except Exception as e:
                # recursive likely blow up
                print(f"Error building tree for QA ID {qa_id}: {e}")
                continue



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset.")
    parser.add_argument("--split_name", type=str, required=True, help="Name of the split (e.g., train, val).")
    parser.add_argument("--cutoff", type=float, default=0.8, help="Cosine similarity cutoff for clustering steps.")
    parser.add_argument("--sentence_transformer_model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Sentence transformer model name for embeddings.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for grading model.")
    parser.add_argument("--build_chains", action="store_true", help="Whether to build chains from trajectories.")
    parser.add_argument("--samples_path", type=str, default="./data/samples/", help="Path to samples directory.")
    parser.add_argument("--processed_path", type=str, default="./data/processed/", help="Path to processed data directory.")
    parser.add_argument("--chains_path", type=str, default="./data/chains/", help="Path to save chains.")
    parser.add_argument("--trees_path", type=str, default="./data/trees/", help="Path to save trees.")
    args = parser.parse_args()

    tree_builder = TrajectoriesToTreeBuilder(
        dataset_name=args.dataset_name,
        split_name=args.split_name,
        sentence_transformer_model_name=args.sentence_transformer_model_name,
        device=args.device,
        samples_path=args.samples_path,
        processed_path=args.processed_path,
        chains_path=args.chains_path,
        trees_path=args.trees_path,
    )

    tree_builder.build_chains_and_trees(
        cutoff=args.cutoff,
        build_chains=args.build_chains,
    )