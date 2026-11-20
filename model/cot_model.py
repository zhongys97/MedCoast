
from model.prm_scorer import PRMScorer
from common.setup_llms import setup_model
from common.call_llms import get_response
from common.prompts import inference_system_prompt


class COTModel:
    def __init__(self,
                 api_keys_json_path: str,
                 llm_model_info: dict,
                    ft_model_path: str,
                 base_model_name: str="answerdotai/ModernBERT-base"):
        
        self.scorer = PRMScorer(api_keys_json_path=api_keys_json_path,
                                base_model_name=base_model_name,
                                 ft_model_path=ft_model_path)
                                 

        self.llm_model_info = llm_model_info

        self.system_prompt = inference_system_prompt


    def _build_messages(self,
                        question: str,
                        steps_so_far: str,
                        step_idx, max_step,
                        must_answer: bool):
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.append({"role": "user", "content": question})
        if steps_so_far.strip():
            messages.append({"role": "assistant", "content": steps_so_far})    

        if not must_answer:
            messages.append({
                "role": "user",
                "content": (
                f"You are at step {step_idx+1}/{max_step}"
                "Output exactly ONE line.\n"
                "If ready: <|answer|> <final answer>\n"
                "Else: <|step|> <next atomic step>\n"
                "Do NOT repeat earlier steps.\n"
                )
            })
        else:
            messages.append({
            "role": "user",
            "content": (
                f"Show your final answer after <|answer|>"
            )
        })

        return messages
    

    def answer_question(self,
                 question: str,
                max_step: int,
                each_step_budget: int,
                accept_threshold: float):

        steps_so_far = ""

        total_num_inference_tokens = 0
        has_answer_flag = False
        for t in range(max_step):  # max steps
            # print(f"=====step {t}========")
            messages = self._build_messages(question, steps_so_far, step_idx=t, max_step=max_step, must_answer=False)
            
            cur_num_try = 0
            while cur_num_try < each_step_budget:
                
                new_step = get_response(model_info=self.llm_model_info,
                            system_prompts=[],
                            user_prompts=[messages],
                            already_messages=True,)[0]["content"]
                
                total_num_inference_tokens += len(self.llm_model_info["tokenizer_instance"](new_step)['input_ids'])
                
                tmp_steps_so_far = steps_so_far + ("\n" if steps_so_far else "") + new_step
                v = self.scorer.score(question, tmp_steps_so_far.split("<|answer|>", 1)[0])   # only score before answer
                if v > accept_threshold or "<|answer|>" in new_step:
                    break
                cur_num_try += 1
            
            steps_so_far += ("\n" if steps_so_far else "") + new_step
            
            if "<|answer|>" in new_step:
                has_answer_flag = True
                break

        if not has_answer_flag:
            messages = self._build_messages(question, steps_so_far, step_idx=t, max_step=max_step, must_answer=True)
            ans = get_response(model_info=self.llm_model_info,
                            system_prompts=[],
                            user_prompts=[messages],
                            already_messages=True,
                            do_sample=False)[0]["content"]
            # print("ans", ans)

            steps_so_far += ("\n" if steps_so_far else "") + ans

        final_ans_token = len(self.llm_model_info["tokenizer_instance"](steps_so_far)['input_ids'])

        return {
            "content": steps_so_far,
            "num_content_tokens": final_ans_token,
            "total_inference_token": total_num_inference_tokens
        }
    

    def answer_question_greedy(self,
                 question: str,
                max_step: int,
                each_step_budget: int,
                accept_threshold: float=None):
        """
        Greedy variant: at each step, sample `each_step_budget` candidates,
        score all, take the best. Stops if the chosen candidate contains <|answer|>.
        Same I/O contract as answer_question().
        """

        steps_so_far = ""
        total_num_inference_tokens = 0
        has_answer_flag = False

        for t in range(max_step):
            messages = self._build_messages(
                question, steps_so_far, step_idx=t, max_step=max_step, must_answer=False
            )

            candidates = []
            answer_candidates = []

            for _ in range(each_step_budget):
                new_step = get_response(
                    model_info=self.llm_model_info,
                    system_prompts=[],
                    user_prompts=[messages],
                    already_messages=True,
                )[0]["content"]

                total_num_inference_tokens += len(
                    self.llm_model_info["tokenizer_instance"](new_step)["input_ids"]
                )

                tmp_steps_so_far = steps_so_far + ("\n" if steps_so_far else "") + new_step
                # only score content before <|answer|>
                prefix = tmp_steps_so_far.split("<|answer|>", 1)[0]
                v = self.scorer.score(question, prefix)

                item = (v, new_step)
                candidates.append(item)
                if "<|answer|>" in new_step:
                    answer_candidates.append(item)

            # Greedy choice:
            # If any candidate answered, choose the best-scoring answer and stop.
            if answer_candidates:
                best_v, best_step = max(answer_candidates, key=lambda x: x[0])
                steps_so_far += ("\n" if steps_so_far else "") + best_step
                has_answer_flag = True
                break

            # Otherwise choose best-scoring step
            best_v, best_step = max(candidates, key=lambda x: x[0])
            steps_so_far += ("\n" if steps_so_far else "") + best_step


        if not has_answer_flag:
            messages = self._build_messages(
                question, steps_so_far, step_idx=t, max_step=max_step, must_answer=True
            )
            ans = get_response(
                model_info=self.llm_model_info,
                system_prompts=[],
                user_prompts=[messages],
                already_messages=True,
                do_sample=False,
            )[0]["content"]
            steps_so_far += ("\n" if steps_so_far else "") + ans

        final_ans_token = len(self.llm_model_info["tokenizer_instance"](steps_so_far)["input_ids"])

        return {
            "content": steps_so_far,
            "num_content_tokens": final_ans_token,
            "total_inference_token": total_num_inference_tokens
        }


    def answer_question_beam(self,
                             question: str,
                             max_step: int,
                             beam_width: int,
                             candidates_per_beam: int) -> dict:
        """
        Beam search variant: at each step, expand every active beam with
        `candidates_per_beam` candidates (sampled stochastically for diversity),
        score all extensions with the PRM, and keep the top `beam_width` beams.
        Terminal beams (containing <|answer|>) are carried forward unchanged.
        Returns the highest-scoring beam at the end.
        """
        beams = [""]
        total_num_inference_tokens = 0

        def _is_terminal(beam: str) -> bool:
            last_step = beam.split("\n")[-1] if beam else ""
            return "<|answer|>" in last_step

        for t in range(max_step):
            active_set = {i for i, b in enumerate(beams) if not _is_terminal(b)}
            if not active_set:
                break

            # Batch all candidate generations across active beams
            all_messages = []
            msg_to_beam = []
            for i in active_set:
                msgs = self._build_messages(
                    question, beams[i], step_idx=t, max_step=max_step, must_answer=False
                )
                for _ in range(candidates_per_beam):
                    all_messages.append(msgs)
                    msg_to_beam.append(i)

            responses = get_response(
                model_info=self.llm_model_info,
                system_prompts=[],
                user_prompts=all_messages,
                already_messages=True,
                do_sample=True
            )

            # Score each candidate extension
            scored = []
            for resp, beam_idx in zip(responses, msg_to_beam):
                new_step = resp["content"]
                total_num_inference_tokens += len(
                    self.llm_model_info["tokenizer_instance"](new_step)["input_ids"]
                )
                old = beams[beam_idx]
                extended = old + ("\n" if old else "") + new_step
                prefix = extended.split("<|answer|>", 1)[0]
                v = self.scorer.score(question, prefix)
                scored.append((v, extended))

            # Carry terminal beams forward (re-score for ranking)
            for i in range(len(beams)):
                if i not in active_set:
                    prefix = beams[i].split("<|answer|>", 1)[0]
                    v = self.scorer.score(question, prefix)
                    scored.append((v, beams[i]))

            scored.sort(key=lambda x: x[0], reverse=True)
            beams = [b for _, b in scored[:beam_width]]

        # Pick the best beam by final PRM score
        best_beam = max(
            beams,
            key=lambda b: self.scorer.score(question, b.split("<|answer|>", 1)[0])
        )

        # Force answer if best beam never terminated
        if not _is_terminal(best_beam):
            msgs = self._build_messages(
                question, best_beam, step_idx=max_step, max_step=max_step, must_answer=True
            )
            ans = get_response(
                model_info=self.llm_model_info,
                system_prompts=[],
                user_prompts=[msgs],
                already_messages=True,
                do_sample=False,
            )[0]["content"]
            best_beam = best_beam + ("\n" if best_beam else "") + ans

        final_ans_token = len(self.llm_model_info["tokenizer_instance"](best_beam)["input_ids"])

        return {
            "content": best_beam,
            "num_content_tokens": final_ans_token,
            "total_inference_token": total_num_inference_tokens
        }


