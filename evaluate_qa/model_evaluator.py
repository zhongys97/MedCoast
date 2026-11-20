from collections import Counter
from common.setup_llms import setup_model
from common.call_llms import get_response
from common.prompts import frq_system_prompt, mcq_system_prompt
from common.utils import *

from model.mcts_model import MCTSModel
from model.cot_model import COTModel
from model.prm_scorer import PRMScorer
from evaluate_qa.utils import summarize_list_of_dicts

import os
import random
import json


def set_global_seed(seed: int):
    """Seed python, numpy and torch RNGs (vLLM is seeded separately via setup_model)."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class ModelEvaluator:
    def __init__(self,
                 processed_qa_pairs_jsons_dir: str,
                 result_save_dir: str = "./evaluation_results",):

        self.ds_name_to_qa_pairs_paths = {}
        for f in os.listdir(processed_qa_pairs_jsons_dir):
             if f.endswith('.json'):
                ds_name = f.split("_")[0]
                self.ds_name_to_qa_pairs_paths[ds_name] = os.path.join(processed_qa_pairs_jsons_dir, f)

                with open(self.ds_name_to_qa_pairs_paths[ds_name], 'r') as file:
                    qa_pairs = json.load(file)
                    print(f"Loaded {len(qa_pairs)} QA pairs for dataset: {ds_name}")

        self.result_save_dir = result_save_dir
        os.makedirs(self.result_save_dir, exist_ok=True)
        self.seed = None
        self.external_prm = None          # "medprm" / "meds3": use that baseline's reward model instead of ours
        self.vllm_gpu_memory_utilization = 0.8
        self._external_scorer = None

    def _seed_suffix(self) -> str:
        # keep each seed's (and each external reward model's) results separate;
        # no suffix when neither is given (old paths unchanged)
        rm = f"_rm-{self.external_prm}" if self.external_prm else ""
        return rm + (f"_seed{self.seed}" if self.seed is not None else "")

    def _get_external_scorer(self):
        if self._external_scorer is None:
            from baselines.external_prm import ADAPTERS
            self._external_scorer = ADAPTERS[self.external_prm]()
        return self._external_scorer


    def get_system_user_prompts_one_qa_pair(self,
                                   qa_pair: dict):
        if qa_pair["is_mcq"]:
            system_prompt = mcq_system_prompt
            user_prompt = str({
                "question": qa_pair["question"],
                "options": qa_pair["options"]
            })
        else:
            system_prompt = frq_system_prompt
            user_prompt = str({
                "question": qa_pair["question"],
            })
        return system_prompt, user_prompt
        

    def get_answer_one_qa_pair(self,
                               model_info: dict,
                               qa_pair: dict,
                               num_answers_needed: int,
                               batch_size: int):

        system_prompt, user_prompt = self.get_system_user_prompts_one_qa_pair(qa_pair)

        responses = []
        for batch_start in range(0, num_answers_needed, batch_size):
            current_batch_size = min(batch_size, num_answers_needed - batch_start)
            batch_system_prompts = [system_prompt] * current_batch_size
            batch_user_prompts = [user_prompt] * current_batch_size

            batch_responses = get_response(
                model_info=model_info,
                system_prompts=batch_system_prompts,
                user_prompts=batch_user_prompts,
                has_thinking=False
            )

            responses.extend(batch_responses)
        return responses


    @staticmethod
    def _prm_prefix(response: dict) -> str:
        """Reasoning before the answer, in the ' <|step|> ...' format the PRM sees during MCTS."""
        steps = process_single_pred_response(response)["steps"]
        if steps:
            return "".join(" <|step|> " + s for s in steps)
        return response["content"].split("<|answer|>", 1)[0]

    def _count_tokens(self, responses: list) -> int:
        total = 0
        for r in responses:
            if "total_inference_token" in r:
                total += r["total_inference_token"]
            else:
                total += r.get("num_content_tokens", 0)
        return total

    def save_each_qa_grade(self,
                model_name: str,
                configs: list,
                 num_answers_needed: int,
                 batch_size: int,
                 bertscore_device: str = None,
                 mcts_config: dict = None,
                 greedy_config: dict = None,
                 beam_search_config: dict = None,
                 best_of_n_config: dict = None) -> str:
        
        

        for cur_config in configs:

            model_info = setup_model(model_name=model_name, use_vllm=True, seed=self.seed,
                                     gpu_memory_utilization=self.vllm_gpu_memory_utilization)

            if cur_config in ["mcts_direct", "mcts_rollout"]:
                assert mcts_config is not None, "mcts_config must be provided for MCTS evaluation."
                mcts = MCTSModel(
                    api_keys_json_path=mcts_config["api_keys_json_path"],
                    ft_model_path=mcts_config["ft_model_path"],
                    llm_model_info=model_info,
                    scorer=self._get_external_scorer() if self.external_prm else None,
                )
                cur_config_details = f"{cur_config}_sim{mcts_config['simulations']}_maxTreeDepth{mcts_config['max_tree_depth']}_maxSearchDepth{mcts_config['max_search_depth']}_branchK{mcts_config['branch_k']}_cPuct{mcts_config['c_puct']}" + self._seed_suffix()
            elif cur_config == "greedy":
                assert greedy_config is not None, "greedy_config must be provided for Greedy evaluation."

                greedy = COTModel(
                    api_keys_json_path=greedy_config["api_keys_json_path"],
                    ft_model_path=greedy_config["ft_model_path"],
                    llm_model_info=model_info,
                )
                cur_config_details = f"{cur_config}_maxStep{greedy_config['max_step']}_eachStepBudget{greedy_config['each_step_budget']}" + self._seed_suffix()
            elif cur_config == "beam_search":
                assert beam_search_config is not None, "beam_search_config must be provided for beam search evaluation."

                beam_searcher = COTModel(
                    api_keys_json_path=beam_search_config["api_keys_json_path"],
                    ft_model_path=beam_search_config["ft_model_path"],
                    llm_model_info=model_info,
                )
                cur_config_details = f"{cur_config}_maxStep{beam_search_config['max_step']}_beamWidth{beam_search_config['beam_width']}_candidatesPerBeam{beam_search_config['candidates_per_beam']}" + self._seed_suffix()
            elif cur_config == "best_of_n":
                assert best_of_n_config is not None, "best_of_n_config must be provided for best-of-n evaluation."
                if self.external_prm:
                    prm_scorer = None
                else:
                    prm_scorer = PRMScorer(
                        api_keys_json_path=best_of_n_config["api_keys_json_path"],
                        base_model_name=best_of_n_config["base_model_name"],
                        ft_model_path=best_of_n_config["ft_model_path"],
                    )
                cur_config_details = f"{cur_config}_n{num_answers_needed}" + self._seed_suffix()

            else:
                cur_config_details = cur_config + self._seed_suffix()



            print(f"Evaluating model: {model_name} with config: {cur_config}")
            for ds_name, qa_pairs_path in self.ds_name_to_qa_pairs_paths.items():
                with open(qa_pairs_path, 'r') as file:
                    qa_pairs = json.load(file)

                model_ds_config_dir = f"{self.result_save_dir}/{cur_config_details}/{model_name.replace('/', '_')}/{ds_name}"
                os.makedirs(model_ds_config_dir, exist_ok=True)

                # fixed evaluation subset (same questions for every seed / run), capped at 300
                qa_pairs = sorted(qa_pairs, key=lambda x: x["idx"])
                if len(qa_pairs) > 300:
                    qa_pairs = random.Random(0).sample(qa_pairs, 300)
                # shuffle processing order for parallelized evaluation; uses its own unseeded RNG
                # so parallel jobs with the same --seed don't walk the same order
                random.Random().shuffle(qa_pairs)

                for qa_pair in qa_pairs:

                    json_save_path = f"{model_ds_config_dir}/qa_{qa_pair['idx']}.json"
                    
                    if os.path.exists(json_save_path):
                        print(f"Skipping already evaluated QA pair idx {qa_pair['idx']}")
                        continue

                    try:
                        if self.external_prm:
                            self._get_external_scorer().set_question(qa_pair, ds_name)
                        ext_rank = None

                        if cur_config == "one_pass":
                            responses = self.get_answer_one_qa_pair(
                                model_info=model_info,
                                qa_pair=qa_pair,
                                num_answers_needed=1,
                                batch_size=1)
                            gt_answers = [qa_pair["answer"]] # as a list

                        elif cur_config == "self_consistency":
                            responses = self.get_answer_one_qa_pair(
                                model_info=model_info,
                                qa_pair=qa_pair,
                                num_answers_needed=num_answers_needed,
                                batch_size=batch_size)
                            gt_answers = [qa_pair["answer"]] * num_answers_needed

                        elif cur_config == "best_of_n":
                            responses = self.get_answer_one_qa_pair(
                                model_info=model_info,
                                qa_pair=qa_pair,
                                num_answers_needed=num_answers_needed,
                                batch_size=batch_size)
                            gt_answers = [qa_pair["answer"]] * num_answers_needed
                            if self.external_prm:
                                # baseline reward model, scored and selected with the baseline's own rules
                                ext_rank = self._get_external_scorer().rank(
                                    qa_pair, [r["content"] for r in responses], ds_name)
                                prm_scores = ext_rank["solution_scores"]
                            else:
                                # PRM is conditioned on the same str({question, options}) user prompt as MCTS
                                _, user_prompt = self.get_system_user_prompts_one_qa_pair(qa_pair)
                                prm_scores = prm_scorer.score_batch(
                                    [user_prompt] * len(responses),
                                    [self._prm_prefix(r) for r in responses])

                        elif cur_config == "mcts_direct":
                            _, user_prompt = self.get_system_user_prompts_one_qa_pair(qa_pair)
                            responses = [mcts.solve(user_prompt,
                                                   mode="direct",
                                                   simulations=mcts_config["simulations"],
                                                   max_tree_depth=mcts_config["max_tree_depth"],
                                                   max_search_depth=mcts_config["max_search_depth"],
                                                   branch_k=mcts_config["branch_k"],
                                                   c_puct=mcts_config["c_puct"],
                                                   )]
                            gt_answers = [qa_pair["answer"]]

                        elif cur_config == "mcts_rollout":
                            _, user_prompt = self.get_system_user_prompts_one_qa_pair(qa_pair)
                            responses = [mcts.solve(user_prompt,
                                                   mode="rollout",
                                                   simulations=mcts_config["simulations"],
                                                   max_tree_depth=mcts_config["max_tree_depth"],
                                                   max_search_depth=mcts_config["max_search_depth"],
                                                   branch_k=mcts_config["branch_k"],
                                                   c_puct=mcts_config["c_puct"],
                                                   )]
                            gt_answers = [qa_pair["answer"]]

                        elif cur_config == "greedy":
                            _, user_prompt = self.get_system_user_prompts_one_qa_pair(qa_pair)
                            responses = [greedy.answer_question(
                                question=user_prompt,
                                max_step=greedy_config["max_step"],
                                each_step_budget=greedy_config["each_step_budget"],
                                accept_threshold=0.8
                            )]
                            gt_answers = [qa_pair["answer"]]

                        elif cur_config == "beam_search":
                            _, user_prompt = self.get_system_user_prompts_one_qa_pair(qa_pair)
                            responses = [beam_searcher.answer_question_beam(
                                question=user_prompt,
                                max_step=beam_search_config["max_step"],
                                beam_width=beam_search_config["beam_width"],
                                candidates_per_beam=beam_search_config["candidates_per_beam"],
                            )]
                            gt_answers = [qa_pair["answer"]]

                        print(responses)

                        pred_final_answers = []
                        for response in responses:
                            pred_ans_steps = process_single_pred_response(response)
                            pred_final_answers.append(pred_ans_steps["pred_answer"])

                        cur_qa_grade = grade_responses(
                            is_mcq=qa_pair["is_mcq"],
                            predicted_answers=pred_final_answers,
                            ground_truth_answers=gt_answers,
                            device=bertscore_device,
                            skip_bert_score=False,
                            options=qa_pair.get("options"),
                        )

                        if cur_config in ["one_pass", "mcts_direct", "mcts_rollout", "greedy", "beam_search", "stepwise_thresholding"]:
                            overall_grade = cur_qa_grade[0]
                        elif cur_config == "self_consistency":
                            if qa_pair["is_mcq"]:
                                # vote on the extracted option letter so "A", "(A)", "A|]" count as the same vote
                                votes = [extract_mcq_letter(p, qa_pair.get("options")) for p in pred_final_answers]
                                votes = [v for v in votes if v] or pred_final_answers
                            else:
                                votes = pred_final_answers
                            most_common_answer, count = Counter(votes).most_common(1)[0]
                            overall_grade = grade_responses(
                                is_mcq=qa_pair["is_mcq"],
                                predicted_answers=[most_common_answer],
                                ground_truth_answers=[gt_answers[0]],
                                device=bertscore_device,
                                skip_bert_score=False,
                                options=qa_pair.get("options"),
                            )[0]
                        elif cur_config == "best_of_n":
                            if ext_rank is not None:
                                best_idx = ext_rank["best_idx"]
                            else:
                                # prefer candidates that produced an answer; among them take the highest PRM score
                                best_idx = max(range(len(responses)),
                                               key=lambda i: (pred_final_answers[i] != "", prm_scores[i]))
                            overall_grade = cur_qa_grade[best_idx]


                        total_tokens = self._count_tokens(responses)

                        responses_with_grades = []
                        for i, response in enumerate(responses):
                            entry = {
                                "qa_pair": qa_pair,
                                "response": response,
                                "pred_answer": pred_final_answers[i],
                                "grade": cur_qa_grade[i]
                            }
                            if cur_config == "best_of_n":
                                entry["prm_score"] = prm_scores[i]
                            responses_with_grades.append(entry)

                        result = {
                            "overall": overall_grade,
                            "total_tokens": total_tokens,
                            "responses_with_grades": responses_with_grades
                        }
                        if cur_config == "best_of_n":
                            result["best_idx"] = best_idx
                            if ext_rank is not None:
                                result["prm"] = self.external_prm
                                result["overall_weighted_vote"] = cur_qa_grade[ext_rank["weighted_vote_idx"]]
                                result["weighted_vote_idx"] = ext_rank["weighted_vote_idx"]
                                result["step_scores"] = ext_rank["step_scores"]
                        with open(json_save_path, 'w') as out_file:
                            json.dump(result, out_file, indent=4)
                    
                    except Exception as e:
                        print(f"Error processing QA pair idx {qa_pair['idx']}: {e}")
                        continue


    def evaluate(self,
                    model_name: str,
                    configs: list,
                    num_answers_needed: int = 5,
                    batch_size: int = 5,
                    bertscore_device: str = None,
                    ft_bert_model_path: str = "./trained_value_model/cutoff_0.8_wcorrect_0.5/checkpoint-22000",
                    mcts_simulation: int = 50,
                    mcts_max_tree_depth: int = 20,
                    mcts_max_search_depth: int = 5,
                    mcts_branch_k: int = 5,
                    mcts_c_puct: float = 1.2,
                    greedy_max_step: int = 5,
                    greedy_each_step_candidate_num: int = 5,
                    beam_search_max_step: int = 12,
                    beam_search_beam_width: int = 4,
                    beam_search_candidates_per_beam: int = 3,
                    seed: int = None,
                    prm_base_model_name: str = "answerdotai/ModernBERT-base",
                    external_prm: str = None,
                    vllm_gpu_memory_utilization: float = None,
                    ) -> str:
            """
            Evaluate the model on the QA pairs and save the results.
            seed: if given, seeds python/numpy/torch/vLLM and saves results under <config>_seed<seed>.
            """

            self.seed = seed
            if seed is not None:
                set_global_seed(seed)
            self.external_prm = external_prm if external_prm not in (None, "none") else None
            if self.external_prm:
                assert set(configs) <= {"best_of_n", "mcts_direct", "mcts_rollout"}, \
                    "--external_prm only applies to best_of_n / mcts_direct / mcts_rollout"
            # an 8B baseline PRM shares the GPU with vLLM -> give vLLM less memory by default
            self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization or (0.55 if self.external_prm else 0.8)

            for item in configs:
                assert item in ["one_pass", "self_consistency", "best_of_n", "mcts_direct", "mcts_rollout", "greedy", "beam_search"], f"Invalid config item: {item}"


            for cur_config in configs:

                if cur_config in ["mcts_direct", "mcts_rollout"]:
                    mcts_config = {
                        "api_keys_json_path": "./api_keys.json",
                        "ft_model_path": ft_bert_model_path,
                        "simulations": mcts_simulation,
                        "max_tree_depth": mcts_max_tree_depth,
                        "max_search_depth": mcts_max_search_depth,
                        "branch_k": mcts_branch_k,
                        "c_puct": mcts_c_puct
                    }
                    greedy_config = None
                    beam_search_config = None
                    best_of_n_config = None
                    cur_config_details = f"{cur_config}_sim{mcts_simulation}_maxTreeDepth{mcts_max_tree_depth}_maxSearchDepth{mcts_max_search_depth}_branchK{mcts_branch_k}_cPuct{mcts_c_puct}" + self._seed_suffix()
                elif cur_config == "greedy":
                    greedy_config = {
                        "api_keys_json_path": "./api_keys.json",
                        "ft_model_path": ft_bert_model_path,
                        "max_step": greedy_max_step,
                        "each_step_budget": greedy_each_step_candidate_num,
                    }
                    mcts_config = None
                    beam_search_config = None
                    best_of_n_config = None
                    cur_config_details = f"{cur_config}_maxStep{greedy_max_step}_eachStepBudget{greedy_each_step_candidate_num}" + self._seed_suffix()
                elif cur_config == "beam_search":
                    beam_search_config = {
                        "api_keys_json_path": "./api_keys.json",
                        "ft_model_path": ft_bert_model_path,
                        "max_step": beam_search_max_step,
                        "beam_width": beam_search_beam_width,
                        "candidates_per_beam": beam_search_candidates_per_beam,
                    }
                    mcts_config = None
                    greedy_config = None
                    best_of_n_config = None
                    cur_config_details = f"{cur_config}_maxStep{beam_search_max_step}_beamWidth{beam_search_beam_width}_candidatesPerBeam{beam_search_candidates_per_beam}" + self._seed_suffix()
                elif cur_config == "best_of_n":
                    best_of_n_config = {
                        "api_keys_json_path": "./api_keys.json",
                        "ft_model_path": ft_bert_model_path,
                        "base_model_name": prm_base_model_name,
                    }
                    mcts_config = None
                    greedy_config = None
                    beam_search_config = None
                    cur_config_details = f"{cur_config}_n{num_answers_needed}" + self._seed_suffix()
                else:
                    mcts_config = None
                    greedy_config = None
                    beam_search_config = None
                    best_of_n_config = None
                    cur_config_details = cur_config + self._seed_suffix()

                self.save_each_qa_grade(
                    model_name=model_name,
                    configs=[cur_config],
                    num_answers_needed=num_answers_needed,
                    batch_size=batch_size,
                    bertscore_device=bertscore_device,
                    mcts_config=mcts_config,
                    greedy_config=greedy_config,
                    beam_search_config=beam_search_config,
                    best_of_n_config=best_of_n_config,
                )

                perf_across_datasets = {}
                all_token_counts = []
                for ds_name, _ in self.ds_name_to_qa_pairs_paths.items():
                    model_ds_config_dir = f"{self.result_save_dir}/{cur_config_details}/{model_name.replace('/', '_')}/{ds_name}"
                    if not os.path.exists(model_ds_config_dir):
                        print(f"No results found for dataset {ds_name} with config {cur_config_details}. Skipping.")
                        continue
                    pred_answers_all_paths = [os.path.join(model_ds_config_dir, f) for f in os.listdir(model_ds_config_dir) if f.endswith('.json')]
                    all_overall_grades = []
                    ds_token_counts = []
                    for p in pred_answers_all_paths:
                        try:
                            with open(p, 'r') as file:
                                data = json.load(file)
                            row = dict(data["overall"])
                            if "overall_weighted_vote" in data:
                                row["weighted_vote"] = data["overall_weighted_vote"]
                            all_overall_grades.append(row)
                            if "total_tokens" in data:
                                ds_token_counts.append(data["total_tokens"])
                        except Exception as e:
                            print(f"Error reading {p}: {e}")
                            continue
                    summarized_results = summarize_list_of_dicts(all_overall_grades)
                    if ds_token_counts:
                        summarized_results["avg_tokens"] = sum(ds_token_counts) / len(ds_token_counts)
                    all_token_counts.extend(ds_token_counts)
                    perf_across_datasets[ds_name] = summarized_results

                if all_token_counts:
                    perf_across_datasets["avg_tokens_all_datasets"] = sum(all_token_counts) / len(all_token_counts)

                # save perf_across_datasets
                perf_save_path = f"{self.result_save_dir}/{cur_config_details}/{model_name.replace('/', '_')}/performance_summary.json"
                os.makedirs(os.path.dirname(perf_save_path), exist_ok=True)
                with open(perf_save_path, 'w') as out_file:
                    json.dump(perf_across_datasets, out_file, indent=4)

                                
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to evaluate.")
    parser.add_argument("--processed_qa_pairs_jsons_dir", type=str, required=True, help="Directory containing processed QA pairs in JSON format.")
    parser.add_argument("--result_save_dir", type=str, default="evaluation_results", help="Directory to save evaluation results.")
    parser.add_argument("--configs", type=str, nargs='+', default=["one_pass", "self_consistency"], help="Evaluation configurations to use.")
    parser.add_argument("--num_answers_needed", type=int, default=20, help="Number of sampled answers for self-consistency / best-of-n.")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for generating answers.")
    parser.add_argument("--bertscore_device", type=str, default=None, help="Device to use for BERTScore computation.")
    parser.add_argument("--ft_bert_model_path", type=str, default="./trained_value_model/cutoff_0.8_wcorrect_0.5/checkpoint-22000", help="Path to the fine-tuned BERT model.")
    parser.add_argument("--mcts_simulation", type=int, default=50, help="Number of MCTS simulations.")
    parser.add_argument("--mcts_max_tree_depth", type=int, default=20, help="Maximum tree depth for MCTS.")
    parser.add_argument("--mcts_max_search_depth", type=int, default=20, help="Maximum search depth for MCTS.")
    parser.add_argument("--mcts_branch_k", type=int, default=5, help="Branching factor k for MCTS.")
    parser.add_argument("--mcts_c_puct", type=float, default=1.2, help="c_puct parameter for MCTS.")
    parser.add_argument("--greedy_max_step", type=int, default=5, help="Maximum steps for greedy CoT.")
    parser.add_argument("--greedy_each_step_candidate_num", type=int, default=5, help="Number of candidates to consider at each step for greedy CoT.")
    parser.add_argument("--beam_search_max_step", type=int, default=5, help="Maximum steps for beam search.")
    parser.add_argument("--beam_search_beam_width", type=int, default=4, help="Number of beams to keep at each step.")
    parser.add_argument("--beam_search_candidates_per_beam", type=int, default=3, help="Number of candidates generated per beam at each step.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (python/numpy/torch/vLLM). Results are saved under <config>_seed<seed>. Omit to keep old behavior and paths.")
    parser.add_argument("--prm_base_model_name", type=str, default="answerdotai/ModernBERT-base", help="Base encoder of the PRM (used by best_of_n).")
    parser.add_argument("--external_prm", type=str, default="none", choices=["none", "medprm", "meds3"], help="Use a baseline reward model (Med-PRM / MedS3, see baselines/) instead of ours for best_of_n / mcts_*. Results get an _rm-<name> suffix.")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=None, help="vLLM GPU memory fraction (default 0.8, or 0.55 with --external_prm).")
    args = parser.parse_args()


    evaluator = ModelEvaluator(
                                processed_qa_pairs_jsons_dir=args.processed_qa_pairs_jsons_dir,
                                result_save_dir=args.result_save_dir
                               )
    
    
    evaluator.evaluate(
        model_name=args.model_name,
        configs=args.configs,
        num_answers_needed=args.num_answers_needed,
        batch_size=args.batch_size,
        bertscore_device=args.bertscore_device,
        ft_bert_model_path=args.ft_bert_model_path,
        mcts_simulation=args.mcts_simulation,
        mcts_max_tree_depth=args.mcts_max_tree_depth,
        mcts_max_search_depth=args.mcts_max_search_depth,
        mcts_branch_k=args.mcts_branch_k,
        mcts_c_puct=args.mcts_c_puct,
        greedy_max_step=args.greedy_max_step,
        greedy_each_step_candidate_num=args.greedy_each_step_candidate_num,
        beam_search_max_step=args.beam_search_max_step,
        beam_search_beam_width=args.beam_search_beam_width,
        beam_search_candidates_per_beam=args.beam_search_candidates_per_beam,
        seed=args.seed,
        prm_base_model_name=args.prm_base_model_name,
        external_prm=args.external_prm,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )