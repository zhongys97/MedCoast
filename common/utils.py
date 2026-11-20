import re

from common.bert_rouge_scores import calculate_bertscore_rougescore


def count_tokens(tokenizer, text: str):
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False).input_ids)


# Answer tag, tolerating malformed variants small models emit: "<|answer|>", "<|answer|", "|answer|>".
_ANSWER_TAG_RE = re.compile(r"<?\|\s*answer\s*\|>?", re.IGNORECASE)
# Anything that ends an answer segment: another tag ("<|...", "|step|") or a stray "|".
_SEGMENT_END_RE = re.compile(r"<\||\|\s*step\s*\||\|", re.IGNORECASE)


def _clean_answer_segment(segment: str) -> str:
    segment = _SEGMENT_END_RE.split(segment, maxsplit=1)[0]
    return segment.strip().strip("*`:").strip()


def process_single_pred_response(response: str) -> str:
    """
    Extract the final answer and reasoning steps from a response.

    The answer is the last NON-EMPTY segment after an answer tag, cut at the next tag / "|".
    Handles outputs like "<|answer|>A|]", "<|answer|>C|.<|answer|>", "<|answer|>A|<|step|> ...",
    and "<|answer|<|step|> ... <|answer|>A". Returns "" if there is no answer tag.
    """

    content = response["content"]

    tags = list(_ANSWER_TAG_RE.finditer(content))
    pred_answer = ""
    answer_start = None
    for m in reversed(tags):
        cleaned = _clean_answer_segment(content[m.end():])
        if cleaned:
            pred_answer = cleaned
            answer_start = m.start()
            break
    if answer_start is None and tags:
        answer_start = tags[-1].start()

    if answer_start is not None:
        reasoning = content[:answer_start].strip()
        steps = [step.strip() for step in reasoning.split("<|step|>") if step.strip()]
    else:
        steps = []

    return {
        "pred_answer": pred_answer,
        "steps": steps
    }


def _norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def extract_mcq_letter(pred_answer: str, options: dict = None) -> str:
    """
    Map a predicted MCQ answer to an option letter. Returns "" if none can be found.

    Tries, in order:
      1. a leading letter: "C", "(C)", "C.", "C) text", "c"
      2. "answer is C" / "option C" / "choice: (C)"
      3. the answer text equal to an option's text, e.g. "yes" -> "A" for PubMedQA
      4. a standalone capital option letter anywhere
    """
    s = (pred_answer or "").strip()
    if not s:
        return ""

    if options and all(len(k) == 1 and k.isalpha() for k in options):
        letters = "".join(sorted(k.upper() for k in options))
    else:
        letters = "ABCDEFGHIJ"

    m = re.match(rf"^[\s\(\[\*]*([{letters}])(?![A-Za-z])", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(rf"\b(?:answer|option|choice)\s*(?:is|:)?\s*[\(\[]?([{letters}])(?![A-Za-z])", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    if options:
        ns = _norm_text(s)
        for k, v in options.items():
            if ns and ns == _norm_text(str(v)):
                return k.upper()

    m = re.search(rf"(?<![A-Za-z])([{letters}])(?![A-Za-z])", s)
    if m:
        return m.group(1)

    return ""


def grade_responses(is_mcq: bool,
                   predicted_answers: list,
                   ground_truth_answers: list,
                   device: str = None,
                   skip_bert_score=True,
                   options: dict = None) -> list:
    """Grades the predicted answer against the ground truth answer.

    Args:
        is_mcq (bool): Whether the question is multiple-choice.
        predicted_answers (list): The answers predicted by the model.
        ground_truth_answers (list): The correct answers.
        options (dict, optional): MCQ options {"A": text, ...}; lets text answers like "yes" map to a letter.

    Returns:
        if is mcq: list of dict: Whether each predicted answer is correct.
        else: list of dict: BERTScore and ROUGE scores for each (pred, ref) pair.
    """
    
    if is_mcq:
        res = []
        for predicted_answer, ground_truth_answer in zip(predicted_answers, ground_truth_answers):
            # For MCQ, compare option letters (A, B, C, D, etc.)
            predicted_letter = extract_mcq_letter(predicted_answer, options)
            ground_truth_answer = ground_truth_answer.upper()[0]
            if not predicted_letter or predicted_letter != ground_truth_answer:
                res.append({"correct": False})
            else:
                res.append({"correct": True})
        return res

    else:
        # For open-ended questions, return bert rouge scores
        scores = calculate_bertscore_rougescore(
            preds=predicted_answers,
            refs=ground_truth_answers,
            batch_size=8,
            device=device,
            skip_bert_score=skip_bert_score,
            rouge_types=("rougeL",)
        )
        return scores

        
if __name__ == "__main__":
    preds = ["The cat sat on the mat.", "A quick brown fox jumps over the dog."]
    refs  = ["A cat is sitting on a mat.", "The quick brown fox jumped over a lazy dog."]
    # out = grade_response(False, preds, refs)
    out = grade_responses(True, ["A x()", "B", "A  c"], {"A", "C", "D"})
    print(out)