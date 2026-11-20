import json
import torch
import sys
import os

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

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from vllm import LLM


def setup_model(model_name: str,
                use_vllm: bool = True,
                seed: int = None,
                gpu_memory_utilization: float = 0.8):
    """
    Set up the model and tokenizer for inference.
    seed: if given, passed to vLLM so sampling is reproducible; if None, vLLM's default is kept.
    gpu_memory_utilization: vLLM's GPU memory fraction; lower it when another large model (e.g. an 8B PRM) shares the GPU.
    """
    seed_kwargs = {"seed": seed} if seed is not None else {}

    if model_name.startswith("gpt-"):
        from openai import OpenAI
        model = OpenAI()
        tokenizer = None

    elif model_name.startswith("gemini-"):
        from google import genai
        model = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        tokenizer = None

    elif model_name.startswith("claude-"):
        import anthropic
        model = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        tokenizer = None


    elif "qwen" in model_name.lower():

        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  trust_remote_code=True)

        if use_vllm:
            model = LLM(
            model=model_name,
            **seed_kwargs,
            quantization="bitsandbytes",  # For 4-bit quantization
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,  # Adjust based on your GPU memory
            trust_remote_code=True,
            enable_prefix_caching=True,
        )

        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 if on A100/H100
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"  # nf4 is recommended
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=quantization_config,
                attn_implementation="flash_attention_2",
                trust_remote_code=True
            ).eval()


    elif "deepseek" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if use_vllm:
            model = LLM(
            model=model_name,
            **seed_kwargs,
            quantization="bitsandbytes",  # For 4-bit quantization
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,  # Adjust based on your GPU memory
            trust_remote_code=True
        )
        else:

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 if on A100/H100
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"  # nf4 is recommended
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            ).eval()

    elif ("llama" in model_name.lower()) or ("meditron" in model_name.lower()):

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if use_vllm:
            model = LLM(
            model=model_name,
            **seed_kwargs,
            quantization="bitsandbytes",  # For 4-bit quantization
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,  # Adjust based on your GPU memory
            enable_prefix_caching=True,
        )
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2"
            ).eval()

    elif "gemma" in model_name.lower():

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if use_vllm:
            # Gemma-2 uses tanh logit softcapping, which vLLM's FlashAttention-3 build (default on H100/H200)
            # does not support. Use the Triton attention backend instead (override by setting the env var).
            os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN_VLLM_V1")
            model = LLM(
            model=model_name,
            **seed_kwargs,
            # quantization="bitsandbytes",  # For 4-bit quantization
            # dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,  # Adjust based on your GPU memory
            max_model_len=8192,
            enable_prefix_caching=True,
        )
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2"
            ).eval()

    elif "allenai" in model_name.lower():

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if use_vllm:
            model = LLM(
            model=model_name,
            **seed_kwargs,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=8192,
            enable_prefix_caching=True,
            trust_remote_code=True,
        )
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2"
            ).eval()

    elif "phi" in model_name.lower():

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if use_vllm:
            model = LLM(
            model=model_name,
            **seed_kwargs,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=8192,
            enable_prefix_caching=True,
            trust_remote_code=True,
        )
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2"
            ).eval()

    elif ("mistral" in model_name.lower()):
        
        is_ministral3 = ("ministral" in model_name.lower()) or ("-2512" in model_name.lower())

        if use_vllm:
            tokenizer = None

            if is_ministral3:
                model = LLM(
                model=model_name,
                **seed_kwargs,
                tokenizer_mode="mistral",
                config_format="mistral",
                load_format="mistral",
                # quantization="bitsandbytes",  # For 4-bit quantization
                dtype="bfloat16",
                gpu_memory_utilization=gpu_memory_utilization,  # Adjust based on your GPU memory
                enable_prefix_caching=True,
            )
            else:
                model = LLM(
                model=model_name,
                **seed_kwargs,
                dtype="bfloat16",
                gpu_memory_utilization=gpu_memory_utilization,
                enable_prefix_caching=True,
            )
        else:
            if is_ministral3:
                from transformers import MistralCommonBackend
                tokenizer = MistralCommonBackend.from_pretrained(model_name)

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )

                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2"
                ).eval()
            
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )

                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2"
                ).eval()

    return {
            "is_vllm": use_vllm,
            "model_name": model_name,
            "model_instance": model,
            "tokenizer_instance": tokenizer
        }
    
