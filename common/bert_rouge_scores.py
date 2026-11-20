import os
import json
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
from rouge_score import rouge_scorer

from typing import List, Dict, Any, Optional, Union



def calculate_bertscore_rougescore(
    
    preds: Union[List[str], str],
    refs: Union[List[str], str],
    device: Optional[str] = None,
    batch_size: int = 16,
    rouge_types: tuple = ("rouge1", "rouge2", "rougeL"),
    skip_bert_score=True
) -> List[Dict[str, Any]]:
    """
    Compute per-example BERTScore and ROUGE.
    Returns: List[Dict], one entry per (pred, ref).
    """
    # Normalize inputs
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(refs, str):
        refs = [refs]
    if len(preds) != len(refs):
        raise ValueError(f"preds and refs must have same length. Got {len(preds)} vs {len(refs)}")
    

    if not skip_bert_score:
        # calculate bert score
        api_keys_json_path = os.path.join(parent_dir, 'api_keys.json')
        with open(api_keys_json_path) as f:
            data = json.load(f)
            os.environ["OPENAI_API_KEY"] = data["OPENAI_API_KEY"]
            if "HF_HOME" in data.keys():
                os.environ["HF_HOME"] = os.path.expanduser(data["HF_HOME"])
            os.environ["HUGGINGFACE_HUB_TOKEN"] = data["HUGGINGFACE_HUB_TOKEN"]

        from bert_score import score as bert_score

        bert_model = "distilroberta-base"

        # -------- BERTScore --------
        P, R, F1 = bert_score(
            cands=preds,
            refs=refs,
            model_type=bert_model,
            idf=False,
            device=device,
            batch_size=batch_size,
            verbose=False,
            rescale_with_baseline=False,
        )

    # -------- ROUGE --------
    scorer = rouge_scorer.RougeScorer(list(rouge_types), use_stemmer=True)

    results: List[Dict[str, Any]] = []

    for i, (pred, ref) in enumerate(zip(preds, refs)):
        if not isinstance(pred, str):
            pred = ""
        if not isinstance(ref, str):
            ref = ""

        rouge_scores = scorer.score(ref, pred)

        if skip_bert_score:
            results.append({
                "rouge": {
                    rt: {
                        "precision": float(rouge_scores[rt].precision),
                        "recall": float(rouge_scores[rt].recall),
                        "f1": float(rouge_scores[rt].fmeasure),
                    }
                    for rt in rouge_types
                },
            })
        else:
            results.append({
                "bertscore": {
                    "precision": float(P[i]),
                    "recall": float(R[i]),
                    "f1": float(F1[i]),
                    "model": bert_model,
                },
                "rouge": {
                    rt: {
                        "precision": float(rouge_scores[rt].precision),
                        "recall": float(rouge_scores[rt].recall),
                        "f1": float(rouge_scores[rt].fmeasure),
                    }
                    for rt in rouge_types
                },
            })

    return results


# Example
if __name__ == "__main__":
    preds = ["The cat sat on the mat.", "A quick brown fox jumps over the dog."]
    refs  = ["A cat is sitting on a mat.", "The quick brown fox jumped over a lazy dog."]
    out = calculate_bertscore_rougescore(preds, refs, batch_size=8, rouge_types=("rougeL",))

    print(out)


