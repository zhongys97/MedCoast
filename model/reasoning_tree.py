from typing import Callable, Dict, Hashable, List, Optional, Tuple
State = Hashable 
import random
import os
import json
from model.prm_scorer import PRMScorer
from common.setup_llms import setup_llm


class ReasoningState:
    def __init__(self,
                 accumulated_resp: str,
                 user_prompt: str):
        self.accumulated_resp: str = accumulated_resp
        self.user_prompt: str = user_prompt


class ReasoningTree:

    def __init__(self,
                 api_keys_json_path: str,
                 ft_model_path: str,
                 base_llm_name: str):
        
        self.scorer = PRMScorer(api_keys_json_path=api_keys_json_path,
                                 ft_model_path=ft_model_path)

        self.llm_model_info = setup_llm(base_llm_name)

        # self.


    def children(self, s: ReasoningState) -> List[ReasoningState]:
        """Return child states of s."""
        raise NotImplementedError

    def is_terminal(self, s: ReasoningState) -> bool:
        """Return True if s is terminal (no further steps / leaf)."""
        raise NotImplementedError

    def prior(self, s: ReasoningState, child: ReasoningState) -> float:
        """
        Optional prior P(s,child) used by PUCT.
        If you don't have priors, return uniform.
        """
        ch = self.children(s)
        return 1.0 / max(1, len(ch))

    def terminal_reward(self, s: ReasoningState) -> float:
        """
        Reward for a terminal state (proxy reward at inference, since no GT).
        E.g., PRM score of completed solution, or stored leaf stats, etc.
        Must be in a consistent range (commonly [0,1]).
        """
        final_before_ans = s["accumulated_resp"].rsplit("<|answer|>", 1)[0]
        question = s["user_prompt"] # the question + options for mcq and question for frq
        score = self.scorer.score(question, final_before_ans)
        # clip between 0 and 1
        score = max(0.0, min(1.0, score))
        return score

    def rollout_policy(self, s: ReasoningState) -> Optional[ReasoningState]:
        """
        Policy used during rollout MCTS: choose a next child from s.
        Return next state or None if no move.
        Default: random child.
        """
        ch = self.children(s)
        if not ch:
            return None
        return random.choice(ch)

