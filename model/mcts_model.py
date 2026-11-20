
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple
from model.prm_scorer import PRMScorer

from common.call_llms import get_response
from common.prompts import inference_system_prompt, inference_followup_prompt_not_must_answer, inference_followup_prompt_must_answer 


class MCTSModel:
    """
    MCTS where the search tree is built on-the-fly by querying the LLM at expansion time.
    State is a tuple of committed step strings: s = (step1, step2, ..., stept)

    Two modes:
      - direct_eval: leaf value = critic(Q, prefix)
      - rollout: leaf value = rollout to terminal, terminal reward via terminal_reward_fn
    """

    def __init__(
        self,
        api_keys_json_path,
        ft_model_path,
        llm_model_info,
        simulations=50,
        max_tree_depth=20,
        max_search_depth=5,
        max_depth=None,
        branch_k=6,
        c_puct=1.2,
        choose_by="visits",
        terminal_reward_fn = None,
        prm_base_model_name="answerdotai/ModernBERT-base",
        scorer=None,
    ):
        # scorer: any object with .score(Q, prefix) -> float (e.g. baselines.external_prm adapters); default is our PRM
        self.scorer = scorer if scorer is not None else PRMScorer(api_keys_json_path=api_keys_json_path,
                                base_model_name=prm_base_model_name,
                                 ft_model_path=ft_model_path)

        self.llm_model_info = llm_model_info

        self.system_prompt = inference_system_prompt

        self.inference_followup_prompt_not_must_answer = inference_followup_prompt_not_must_answer
        self.inference_followup_prompt_must_answer = inference_followup_prompt_must_answer
        
        self.simulations = simulations
        # Depth controls
        # - max_tree_depth: maximum depth explored inside each MCTS simulation
        # - max_search_depth: maximum number of committed steps in the final reasoning chain
        if max_depth is not None and max_tree_depth == 5 and max_search_depth == 5:
            max_tree_depth = max_depth
            max_search_depth = max_depth
        self.max_tree_depth = max_tree_depth
        self.max_search_depth = max_search_depth
        self.branch_k = branch_k
        self.c_puct = c_puct
        
        assert choose_by in ["visits", "q"], "visits or q"
        self.choose_by = choose_by
        
        self.terminal_reward_fn = terminal_reward_fn or self._critic

        # Search tree cache: children[state] = list of actions (step strings)
        self.children: Dict[Tuple[str, ...], List[str]] = {}
        self.expanded: Dict[Tuple[str, ...], bool] = {}

        # MCTS stats
        self.N_s: Dict[Tuple[str, ...], int] = {}
        self.N_sa: Dict[Tuple[str, ...], int] = {}     # keyed by (state, action_hash)
        self.W_sa: Dict[Tuple[str, ...], float] = {}   # same key
            
        self.total_num_inference_tokens = 0
            
            
    def _build_messages(self, question: str, steps_so_far: str, must_answer: bool):

        if "llama" in self.llm_model_info["model_name"].lower():
            # has system prompt
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.append({"role": "user", "content": question})
        else:
            # no system prompt
            combined_mesage = self.system_prompt + "\n" + question
            messages = [{"role": "user", "content": combined_mesage}]

        if steps_so_far.strip():
            messages.append({"role": "assistant", "content": steps_so_far})    

            if not must_answer:
                messages.append({
                    "role": "user",
                    "content": self.inference_followup_prompt_not_must_answer
                })
            else:
                messages.append({
                "role": "user",
                "content": self.inference_followup_prompt_must_answer
            })

        return messages
    
    
    def _critic(self, Q: str, prefix) -> float:
        steps_so_far = "".join(prefix)
        return self.scorer.score(Q, steps_so_far)
    
    @staticmethod
    def _is_terminal(Q: str, prefix) -> bool:
        return len(prefix) > 0 and "<|answer|>" in prefix[-1]

    def _propose(self, Q: str, prefix, k: int):

        steps_so_far = "".join(prefix)
        messages = [self._build_messages(Q, steps_so_far, must_answer=False) for _ in range(k)]
        new_steps = get_response(model_info=self.llm_model_info,
                system_prompts=[],
                user_prompts=messages,
                max_tokens=64,
                already_messages=True)
        
        new_steps_str = []
        for candidate in new_steps:
            new_candidate_str = candidate["content"].strip()

            if new_candidate_str.startswith("<|step|>"):
                new_steps_str.append(" <|step|> " + new_candidate_str.split("<|step|>")[1].strip()) # only keep one step
            elif new_candidate_str.startswith("<|answer|>"):
                new_steps_str.append(new_candidate_str.strip())
            else:
                new_steps_str.append(" <|step|> " + new_candidate_str)
            self.total_num_inference_tokens += candidate["num_content_tokens"]
        return new_steps_str


    @staticmethod
    def _edge_key(state: Tuple[str, ...], action: str) -> Tuple[str, ...]:
        # Use a tuple key to avoid collisions; action is included as last element in key.
        return state + ("<A>", action)

    def _Q(self, s: Tuple[str, ...], a: str) -> float:
        k = self._edge_key(s, a)
        n = self.N_sa.get(k, 0)
        return 0.0 if n == 0 else self.W_sa.get(k, 0.0) / n

    def _expand(self, Q: str, s: Tuple[str, ...]) -> None:
        if self.expanded.get(s, False):
            return
        # Generate K candidates and cache
        cand = self._propose(Q, list(s), self.branch_k)
        # basic dedup while preserving order
        seen = set()
        uniq = []
        for x in cand:
            x = x.strip()
            if x and x not in seen:
                uniq.append(x)
                seen.add(x)
        self.children[s] = uniq
        self.expanded[s] = True
        

    def _score_puct(self, s: Tuple[str, ...], a: str) -> float:
        # Uniform prior: P(s,a) = 1/|A(s)| (constant across actions at node)
        # Keeping it for correctness/clarity.
        A = self.children.get(s, [])
        P = 1.0 / max(1, len(A))
        Ns = self.N_s.get(s, 0)
        Nsa = self.N_sa.get(self._edge_key(s, a), 0)
        U = self.c_puct * P * math.sqrt(Ns + 1e-8) / (1.0 + Nsa)
        return self._Q(s, a) + U

    def _select(self, s: Tuple[str, ...]) -> str:
        A = self.children.get(s, [])
        if not A:
            raise RuntimeError("Select called on a node with no children.")
        best_a, best_sc = None, -1e18
        for a in A:
            sc = self._score_puct(s, a)
            if sc > best_sc:
                best_sc = sc
                best_a = a
        assert best_a is not None
        return best_a

    def _backup(self, path: List[Tuple[Tuple[str, ...], str]], v: float) -> None:
        # Update edge stats only; node visit counts are updated during traversal.
        for (s, a) in path:
            ek = self._edge_key(s, a)
            self.N_sa[ek] = self.N_sa.get(ek, 0) + 1
            self.W_sa[ek] = self.W_sa.get(ek, 0.0) + float(v)

    def _final_action(self, root: Tuple[str, ...]) -> Optional[str]:
        A = self.children.get(root, [])
        if not A:
            return None
        if self.choose_by == "q":
            return max(A, key=lambda a: self._Q(root, a))
        else:
            return max(A, key=lambda a: self.N_sa.get(self._edge_key(root, a), 0))


    def mcts_next_step_direct(self, Q: str, prefix_steps: Sequence[str]) -> Optional[str]:
        self._reset_stats_keep_cache()
        root = tuple(prefix_steps)

        # Ensure root expanded (so we can pick a root action at end)
        if not self._is_terminal(Q, list(root)):
            self._expand(Q, root)

        for _ in range(self.simulations):
            s = root
            path: List[Tuple[Tuple[str, ...], str]] = []

            while True:
                # node visit update (common choice)
                self.N_s[s] = self.N_s.get(s, 0) + 1

                if self._is_terminal(Q, list(s)) or len(s) >= self.max_search_depth:
                    v = self.terminal_reward_fn(Q, list(s))
                    break

                if not self.expanded.get(s, False):
                    # Expand on first visit
                    self._expand(Q, s)
                    # Direct evaluation at newly expanded leaf (no rollout)
                    v = self._critic(Q, list(s))
                    break

                if not self.children.get(s, []):
                    v = self._critic(Q, list(s))
                    break

                a = self._select(s)
                path.append((s, a))
                s = s + (a,)

                # Stop immediately when reaching an unexpanded node (leaf of search tree)
                if not self.expanded.get(s, False):
                    self._expand(Q, s)  # optional; makes next visits cheaper
                    v = self._critic(Q, list(s))
                    break

            self._backup(path, v)

        return self._final_action(root)


    def mcts_next_step_rollout(self, Q: str, prefix_steps: Sequence[str]) -> Optional[str]:
        self._reset_stats_keep_cache()
        root = tuple(prefix_steps)

        if not self._is_terminal(Q, list(root)):
            self._expand(Q, root)

        for _ in range(self.simulations):
            s = root
            path: List[Tuple[Tuple[str, ...], str]] = []

            while True:
                self.N_s[s] = self.N_s.get(s, 0) + 1

                if self._is_terminal(Q, list(s)) or len(s) >= self.max_search_depth:
                    v = self.terminal_reward_fn(Q, list(s))
                    break

                if not self.expanded.get(s, False):
                    self._expand(Q, s)
                    # Rollout from this leaf (simulation policy)
                    v = self._rollout(Q, s)
                    break

                if not self.children.get(s, []):
                    v = self.terminal_reward_fn(Q, list(s))
                    break

                a = self._select(s)
                path.append((s, a))
                s = s + (a,)

                if not self.expanded.get(s, False):
                    self._expand(Q, s)
                    v = self._rollout(Q, s)
                    break

            self._backup(path, v)

        return self._final_action(root)

    def _rollout(self, Q: str, start: Tuple[str, ...]) -> float:
        """
        Simple rollout policy: repeatedly expand (if needed) and pick a random child
        until terminal or max depth, then score terminal with terminal_reward_fn.
        """
        s = start
        while not self._is_terminal(Q, list(s)) and len(s) < self.max_search_depth:
            if not self.expanded.get(s, False):
                self._expand(Q, s)
            A = self.children.get(s, [])
            if not A:
                break
            a = random.choice(A)
            s = s + (a,)
        return self.terminal_reward_fn(Q, list(s))

    def _reset_stats_keep_cache(self) -> None:
        # Clear only per-search statistics; keep children cache (optional).
        self.N_s.clear()
        self.N_sa.clear()
        self.W_sa.clear()

    def _reset_all(self) -> None:
        # Clear all statistics and cached expansions.
        self.children.clear()
        self.expanded.clear()
        self._reset_stats_keep_cache()


    def solve(self,
                Q: str,
                mode: str = "direct",
                simulations=50,
                max_tree_depth=5,
                max_search_depth=5,
                max_depth=None,
                branch_k=6,
                c_puct=1.2,) -> List[str]:
        """
        Replans at each committed step from the current prefix.
        Returns the committed reasoning steps (not the final answer string).
        """
        
        self.simulations = simulations
        if max_depth is not None and max_tree_depth == 5 and max_search_depth == 5:
            max_tree_depth = max_depth
            max_search_depth = max_depth
        self.max_tree_depth = max_tree_depth
        self.max_search_depth = max_search_depth
        self.branch_k = branch_k
        self.c_puct = c_puct
        self.total_num_inference_tokens = 0
        self._reset_all()
        
        steps: List[str] = []
        for _ in range(self.max_tree_depth):
            if self._is_terminal(Q, steps):
                break
            if mode == "direct":
                a = self.mcts_next_step_direct(Q, steps)
            elif mode == "rollout":
                a = self.mcts_next_step_rollout(Q, steps)
            else:
                raise ValueError("mode must be 'direct' or 'rollout'")
            if a is None:
                break
            steps.append(a)
            
        reasoning = "".join(steps)

        if len(steps) > 0 and "<|answer|>" not in steps[-1]:
            messages = self._build_messages(Q, reasoning, must_answer=True)
            pred_ans_dict = get_response(model_info=self.llm_model_info,
                        system_prompts=[],
                        user_prompts=[messages],
                        max_tokens=64,
                        already_messages=True)[0]
            self.total_num_inference_tokens += pred_ans_dict["num_content_tokens"]

            res = reasoning + pred_ans_dict["content"]
            final_ans_token = len(self.llm_model_info["tokenizer_instance"](res)['input_ids'])

        else:
            res = reasoning
            final_ans_token = len(self.llm_model_info["tokenizer_instance"](res)['input_ids'])

        if "<|answer|>" not in res:
            messages = self._build_messages(Q, res, must_answer=True)
            pred_ans_dict = get_response(model_info=self.llm_model_info,
                        system_prompts=[],
                        user_prompts=[messages],
                        max_tokens=64,
                        already_messages=True)[0]
            self.total_num_inference_tokens += pred_ans_dict["num_content_tokens"]
            res = res + pred_ans_dict["content"]
            final_ans_token = len(self.llm_model_info["tokenizer_instance"](res)['input_ids'])

        return {
                "content": res,
                "num_content_tokens": final_ans_token,
                "total_inference_token": self.total_num_inference_tokens
            }
            