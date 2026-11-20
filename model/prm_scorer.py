import json
import os
import hashlib
from collections import OrderedDict
from typing import List, Union, Tuple, Optional

import torch
import torch.nn as nn

class PRM(nn.Module):
    """Score (Q, prefix) -> scalar."""
    def __init__(self,
                 api_keys_json_path,
                 base_model_name: str):
        super().__init__()

        with open(api_keys_json_path) as f:
            data = json.load(f)

        if "HF_HOME" in data:
            os.environ["HF_HOME"] = os.path.expanduser(data["HF_HOME"])
        os.environ["HUGGINGFACE_HUB_TOKEN"] = data["HUGGINGFACE_HUB_TOKEN"]

        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(base_model_name)
        hid = self.model.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hid, 1),
        )

    def forward(self, **enc) -> torch.Tensor:
        out = self.model(**enc)
        # robust CLS usage (works even if pooler_output missing)
        cls = out.last_hidden_state[:, 0]
        return self.head(cls).squeeze(-1)

class LRUCache:
    """Tiny LRU cache for scores."""
    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class PRMScorer(nn.Module):
    def __init__(
        self,
        api_keys_json_path: str,
        base_model_name: str,
        ft_model_path: str,
        *,
        device: Optional[str] = None,
        max_length: int = 2048,            # <--- don’t default to 8192
        truncation_side: str = "left",     # keep latest steps
        amp_dtype: str = "bfloat16",       
        cache_size: int = 50000,
        prefix_char_limit: Optional[int] = None,  # e.g., 8000 to avoid huge strings
    ):
        super().__init__()

        # env setup ONCE
        with open(api_keys_json_path) as f:
            data = json.load(f)
        if "HF_HOME" in data:
            os.environ["HF_HOME"] = os.path.expanduser(data["HF_HOME"])
        os.environ["HUGGINGFACE_HUB_TOKEN"] = data["HUGGINGFACE_HUB_TOKEN"]

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        self.tokenizer.truncation_side = truncation_side

        self.prm = PRM(api_keys_json_path=api_keys_json_path,
                       base_model_name=base_model_name)

        state_dict = torch.load(ft_model_path, map_location="cpu")
        self.prm.load_state_dict(state_dict["model_state"], strict=True)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.max_length = int(max_length)
        self.prefix_char_limit = prefix_char_limit

        # AMP dtype
        if amp_dtype == "bfloat16":
            self.amp_dtype = torch.bfloat16
        elif amp_dtype == "float16":
            self.amp_dtype = torch.float16
        else:
            raise ValueError("amp_dtype must be 'bfloat16' or 'float16'")

        self.prm.to(self.device)
        self.prm.eval()
        for p in self.prm.parameters():
            p.requires_grad_(False)

        self.cache = LRUCache(cache_size)

    def _key(self, q: str, p: str) -> str:
        # hash avoids huge keys
        h = hashlib.blake2b((q + "\n" + p).encode("utf-8"), digest_size=16).hexdigest()
        return h

    def _maybe_trim_prefix(self, pred_content: str) -> str:
        if self.prefix_char_limit is None:
            return pred_content
        if len(pred_content) <= self.prefix_char_limit:
            return pred_content
        # keep last part (matches left truncation intent)
        return pred_content[-self.prefix_char_limit:]

    def score_batch(
        self,
        Qs: List[str],
        preds: List[str],
        *,
        return_tensor: bool = False,
    ) -> Union[List[float], torch.Tensor]:
        assert len(Qs) == len(preds)

        # cache lookup
        outs = [None] * len(Qs)
        todo_idx = []
        todo_Q, todo_P, todo_keys = [], [], []

        for i, (q, p) in enumerate(zip(Qs, preds)):
            p = self._maybe_trim_prefix(p)
            k = self._key(q, p)
            v = self.cache.get(k)
            if v is None:
                todo_idx.append(i)
                todo_Q.append(q)
                todo_P.append(p)
                todo_keys.append(k)
            else:
                outs[i] = v

        if todo_idx:
            enc = self.tokenizer(
                todo_Q,
                todo_P,
                padding=True,            # pad to longest in this batch only
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}

            with torch.inference_mode():
                if self.device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                        scores = self.prm(**enc)
                else:
                    scores = self.prm(**enc)

            scores = scores.detach().float().cpu().tolist()
            for i, k, s in zip(todo_idx, todo_keys, scores):
                outs[i] = float(s)
                self.cache.put(k, float(s))

        if return_tensor:
            return torch.tensor(outs, dtype=torch.float)
        return outs

    def score(self, Q: str, pred_content: str) -> float:
        return self.score_batch([Q], [pred_content])[0]
    


class old_PRMScorer:
    def __init__(self, api_keys_json_path, ft_model_path,
                 base_model_name="answerdotai/ModernBERT-base"):

        with open(api_keys_json_path) as f:
            data = json.load(f)

        if "HF_HOME" in data:
            os.environ["HF_HOME"] = os.path.expanduser(data["HF_HOME"])
        os.environ["HUGGINGFACE_HUB_TOKEN"] = data["HUGGINGFACE_HUB_TOKEN"]

        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "right"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # If no GPU, DO NOT use flash-attn
        use_flash = (self.device.type == "cuda")
        attn_impl = "flash_attention_2" if use_flash else "eager"
        dtype = torch.bfloat16 if use_flash else torch.float32

        self.model = AutoModelForSequenceClassification.from_pretrained(
            ft_model_path,
            attn_implementation=attn_impl,
            dtype=dtype,
        ).to(self.device)
        self.model.eval()

    def score(self, user_prompt, pred_content):
        inputs = self.tokenizer(
            user_prompt,
            pred_content,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=8192,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            if self.device.type == "cuda":
                # autocast makes sure ops run bf16/fp16 for flash-attn compatibility
                with torch.autocast(device_type="cuda", dtype=self.model.dtype):
                    logits = self.model(**inputs).logits
            else:
                logits = self.model(**inputs).logits

        return logits.squeeze().item()


if __name__ == "__main__":
    # huggingface-cli download zhongys97/medcoast medcoast_prm_dsim0.8_a0.0.pt --local-dir checkpoints
    scorer = PRMScorer(api_keys_json_path="api_keys.json",
                       base_model_name="answerdotai/ModernBERT-base",
                       ft_model_path="checkpoints/medcoast_prm_dsim0.8_a0.0.pt")
    user_prompt = str({'question': 'A 6-month-old boy is brought to her pediatrician for a checkup and vaccines. He was born at 39 weeks gestation via spontaneous vaginal delivery. He is meeting all the developmental milestones. His prenatal history was complicated with maternal diabetes. He is being breastfed exclusively and has 3–4 wet diapers a day. On physical exam, the vital signs include: temperature 37.0°C (98.6°F), blood pressure 85/45 mm Hg, pulse 140/min, and respiratory rate is 31/min. On physical examination, the patient is alert and responsive. Height, weight, and head circumference are above the 90th percentile. Which of the following vaccines should be administered for this patient at this visit?', 'options': {'A': 'RV, DTaP, PCV', 'B': 'RV, Tdap, PCV, Hib', 'C': 'RV, DTaP, PCV, Hib, IPV', 'D': 'Hep B, DTaP, Influenza'}})
    pred_content = "<|step|> Recall vaccine schedule for 6-month-old infants <|step|> Note that this baby is above the 90th percentile for height, weight, and head circumference, which may indicate macrosomia due to maternal diabetes. <|step|> Consider the potential risks of macrosomia, such as increased risk of respiratory problems. <|step|> Recall that the recommended vaccinations for 6-month-old infants include RotaVirus (RV), Diphtheria, Tetanus, and Pertussis (DTaP), and Pneumococcal Conjugate Vaccine (PCV)."
    score = scorer.score(user_prompt, pred_content)
    print(f"Score: {score}")