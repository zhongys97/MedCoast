import torch
import os
import json
import time
from vllm import SamplingParams
from common.utils import count_tokens


def get_response(model_info: dict,
                 system_prompts: list,
                 user_prompts: list,
                 max_tokens: int = 4096,
                 already_messages: bool = False,
                 has_thinking: bool = False,
                 do_sample: bool = True
                ):
    """
    Get response from the model based on the provided prompt.
    model_info has four keys: is_vllm (bool), model_name (str), model_instance (object), tokenizer_instance (object)
    """
    is_vllm = model_info["is_vllm"]
    model_name = model_info["model_name"]
    model = model_info["model_instance"]
    tokenizer = model_info["tokenizer_instance"]

    if tokenizer is None and is_vllm and hasattr(model, "get_tokenizer"):
        try:
            tokenizer = model.get_tokenizer()
        except Exception:
            pass

    if model_name.startswith("gpt-"):
        responses = []

        for i in range(len(system_prompts)):
            response = model.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": system_prompts[i] + "\n" + user_prompts[i]}],
            )
            responses.append({"content": response.choices[0].message.content})
        return responses
    
    elif model_name.startswith("gemini-"):

        responses = []
        for i in range(len(system_prompts)):
            response = model.models.generate_content(
                model=model_name, 
                contents=str({"role": "user", "content": system_prompts[i] + "\n" + user_prompts[i]})
            )
            responses.append({"content": response.text})

        return responses
    
    elif model_name.startswith("claude-"):

        responses = []
        for i in range(len(system_prompts)):
            response = model.messages.create(
                model=model_name,
                max_tokens=1024,
                messages=[{"role": "user", "content": system_prompts[i] + "\n" + user_prompts[i]}],
            )
            responses.append({"content": response.content[0].text})
        return responses
    
    elif "qwen" in model_name.lower() and "deepseek" not in model_name.lower():

        if already_messages:
            messages = user_prompts
        else:
            messages = []
            for i in range(len(system_prompts)):
                prompt = system_prompts[i] + "\n" + user_prompts[i]
                messages.append([{"role": "user", "content": prompt}])

        def _split_qwen_think(text: str):
            """
            Qwen3 thinking mode typically emits: <think> ... </think> answer...
            Return (thinking, content). If tags missing, thinking="".
            """
            if not text:
                return "", ""
            if "</think>" in text:
                left, right = text.split("</think>", 1)
                thinking = left.replace("<think>", "").strip()
                content = right.strip()
                return thinking, content
            return "", text.strip()

        if is_vllm:

            if do_sample:

                if has_thinking:
                    sampling_params = SamplingParams(temperature=0.6,
                                                    top_p=0.95,
                                                    top_k=20,
                                                    max_tokens=max_tokens,
                                                    repetition_penalty=1.2)
                else:
                    sampling_params = SamplingParams(temperature=0.7,
                                                    top_p=0.8,
                                                    top_k=20,
                                                    max_tokens=max_tokens,
                                                    #  repetition_penalty=1.2
                                                    )
                outputs = model.chat(
                    messages,
                    sampling_params=sampling_params,
                    chat_template_kwargs={"enable_thinking": has_thinking},
                )
            else:
                sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)
                outputs = model.chat(
                    messages,
                    sampling_params=sampling_params,
                    chat_template_kwargs={"enable_thinking": has_thinking},
                )

            response_dicts = []
            for output in outputs:
                text_out = output.outputs[0].text  # vLLM RequestOutput -> first candidate text

                if has_thinking:
                    thinking, content = _split_qwen_think(text_out)
                    response_dicts.append({
                        "thinking": thinking,
                        "content": content,
                        "num_thinking_tokens": count_tokens(tokenizer, thinking),
                        "num_content_tokens": count_tokens(tokenizer, content)
                    })
                else:
                    response_dicts.append({
                        "thinking": "",
                        "content": text_out,
                        "num_thinking_tokens": 0,
                        "num_content_tokens": count_tokens(tokenizer, text_out)
                    })
            return response_dicts
        
        else:
            raise NotImplementedError("Non-vLLM Qwen models are not implemented yet.")

        
    elif "gemma" in model_name.lower():


        if already_messages:
            # Gemma does not support system role — merge system content into the next user message
            messages = []
            for conv in user_prompts:
                new_conv = []
                pending_system = ""
                for msg in conv:
                    if msg["role"] == "system":
                        pending_system += msg["content"] + "\n"
                    elif msg["role"] == "user":
                        content = (pending_system + msg["content"]) if pending_system else msg["content"]
                        new_conv.append({"role": "user", "content": content})
                        pending_system = ""
                    else:
                        new_conv.append(msg)
                messages.append(new_conv)
        else:
            messages = []
            for i in range(len(system_prompts)):
                prompt = system_prompts[i] + "\n" + user_prompts[i]
                messages.append([{"role": "user", "content": prompt}])


        if is_vllm:

            if do_sample:
                sampling_params = SamplingParams(temperature=0.7,
                                                top_p=0.9,
                                                max_tokens=max_tokens,
                                                repetition_penalty=1.2,
                                                )
                
                outputs = model.chat(
                    messages,
                    sampling_params=sampling_params,
                )
            else:
                sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)
                outputs = model.chat(
                    messages,
                    sampling_params=sampling_params,
                )

            response_dicts = []
            for output in outputs:
                text_out = output.outputs[0].text  # vLLM RequestOutput -> first candidate text
                response_dicts.append({
                    "thinking": "",
                    "content": text_out,
                    "num_thinking_tokens": 0,
                    "num_content_tokens": count_tokens(tokenizer, text_out)
                })

            return response_dicts

        else:
            raise NotImplementedError("Non-vLLM Gemma models are not implemented yet.")

    elif ("llama" in model_name.lower()) or ("meditron" in model_name.lower()):

        if already_messages:
            messages = user_prompts
        else:
            messages = []
            for i in range(len(system_prompts)):
                messages.append([{"role": "system", "content":  system_prompts[i]},
                            {"role": "user", "content": user_prompts[i]}])

        if is_vllm:

            if do_sample:
                sampling_params = SamplingParams(temperature=0.7,
                                                top_p=0.9,
                                                max_tokens=max_tokens,
                                                #  repetition_penalty=1.2
                                                )
                outputs = model.chat(
                    messages,
                    sampling_params=sampling_params,
                )
            else:
                sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)
                outputs = model.chat(
                    messages,
                    sampling_params=sampling_params,
                )

            response_dicts = []
            for output in outputs:
                text_out = output.outputs[0].text  # vLLM RequestOutput -> first candidate text
                response_dicts.append({
                    "thinking": "",
                    "content": text_out,
                    "num_thinking_tokens": 0,
                    "num_content_tokens": count_tokens(tokenizer, text_out)
                })

            return response_dicts

        else:
            raise NotImplementedError("Non-vLLM Llama models are not implemented yet.")
    
    elif "allenai" in model_name.lower():

        if already_messages:
            messages = user_prompts
        else:
            messages = []
            for i in range(len(system_prompts)):
                messages.append([{"role": "system", "content": system_prompts[i]},
                                  {"role": "user", "content": user_prompts[i]}])

        if is_vllm:

            if do_sample:
                sampling_params = SamplingParams(temperature=0.7,
                                                top_p=0.9,
                                                max_tokens=max_tokens,
                                                )
                outputs = model.chat(
                    messages,
                    sampling_params=sampling_params,
                )
            else:
                sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)
                outputs = model.chat(
                    messages,
                    sampling_params=sampling_params,
                )

            response_dicts = []
            for output in outputs:
                text_out = output.outputs[0].text
                response_dicts.append({
                    "thinking": "",
                    "content": text_out,
                    "num_thinking_tokens": 0,
                    "num_content_tokens": count_tokens(tokenizer, text_out)
                })

            return response_dicts

        else:
            raise NotImplementedError("Non-vLLM AllenAI models are not implemented yet.")

    elif "phi" in model_name.lower():

        if already_messages:
            messages = user_prompts
        else:
            messages = []
            for i in range(len(system_prompts)):
                messages.append([{"role": "system", "content": system_prompts[i]},
                                  {"role": "user", "content": user_prompts[i]}])

        if is_vllm:

            if do_sample:
                sampling_params = SamplingParams(temperature=0.7,
                                                top_p=0.9,
                                                max_tokens=max_tokens,
                                                repetition_penalty=1.1,
                                                )
                outputs = model.chat(
                    messages,
                    sampling_params=sampling_params,
                )
            else:
                sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)
                outputs = model.chat(
                    messages,
                    sampling_params=sampling_params,
                )

            response_dicts = []
            for output in outputs:
                text_out = output.outputs[0].text
                response_dicts.append({
                    "thinking": "",
                    "content": text_out,
                    "num_thinking_tokens": 0,
                    "num_content_tokens": count_tokens(tokenizer, text_out)
                })

            return response_dicts

        else:
            raise NotImplementedError("Non-vLLM Phi models are not implemented yet.")

    elif "mistral" in model_name.lower():

        if already_messages:
            prompts = []
            for conv in user_prompts:
                text = []
                for m in conv:
                    if m["role"] == "system":
                        text.append(m["content"])
                    elif m["role"] == "user":
                        text.append(m["content"])
                    elif m["role"] == "assistant":
                        text.append(m["content"])
                prompts.append("\n".join(text))
        else:
            prompts = [
                system_prompts[i] + "\n" + user_prompts[i]
                for i in range(len(system_prompts))
            ]

        if is_vllm:

            if do_sample:
                sampling_params = SamplingParams(
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=max_tokens,
                    repetition_penalty=1.2,
                    min_tokens=128,
                )

                outputs = model.generate(
                    prompts,
                    sampling_params=sampling_params,
                )
            else:
                sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)
                outputs = model.generate(
                    prompts,
                    sampling_params=sampling_params,
                )

            response_dicts = []
            for output in outputs:
                text_out = output.outputs[0].text

                n_tok = (
                    count_tokens(tokenizer, text_out)
                    if tokenizer is not None else -1
                )

                response_dicts.append({
                    "thinking": "",
                    "content": text_out,
                    "num_thinking_tokens": 0,
                    "num_content_tokens": n_tok
                })

            return response_dicts

        else:
            raise NotImplementedError("Non-vLLM Mistral models are not implemented yet.")


    

