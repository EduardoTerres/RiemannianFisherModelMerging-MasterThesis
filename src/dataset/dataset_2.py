"""Dataset set 2: extra fine-tuning tasks plus math and code tasks from set 1."""

from __future__ import annotations

import json
from typing import Any, Callable

from datasets import load_dataset
from torch.utils.data import DataLoader


TaskSpec = tuple[str, str, str | None, str, Callable[[dict[str, Any]], tuple[str, str]]]


def _first(doc: dict[str, Any], *keys: str, default: str = "") -> Any:
    for key in keys:
        if key in doc and doc[key] not in (None, "", []):
            return doc[key]
    return default


def _stringify(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(_stringify(v) for v in value[:8])
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)[:4000]
    return str(value)


def expand_doc(task: str, doc: dict[str, Any]) -> list[dict[str, Any]]:
    return [doc]


def _coqa_values(value: Any) -> list[Any]:
    if isinstance(value, dict):
        value = value.get("input_text", [])
    return value if isinstance(value, list) else []


def _coqa_text(doc: dict[str, Any]) -> tuple[str, str]:
    questions = _coqa_values(doc.get("questions", []))
    answers = _coqa_values(doc.get("answers", []))
    prompt = doc.get("story", "") + "\n\n"
    for question, answer in zip(questions, answers[:-1]):
        prompt += f"Q: {question}\n\nA: {answer}\n\n"
    if questions:
        prompt += f"Q: {questions[-1]}\n\nA:"
    target = _stringify(answers[-1]) if answers else ""
    return prompt, prompt + target


def _drop_answer_text(answer: Any) -> str:
    if not isinstance(answer, dict):
        return _stringify(answer)
    number = answer.get("number")
    if number:
        return str(number)
    spans = answer.get("spans")
    if spans:
        return ", ".join(_stringify(span) for span in spans)
    date = answer.get("date", {})
    if isinstance(date, dict):
        return " ".join(
            str(date.get(part, "")).strip()
            for part in ("day", "month", "year")
            if str(date.get(part, "")).strip()
        )
    return _stringify(answer)


def _drop_text(doc: dict[str, Any]) -> tuple[str, str]:
    prompt = f"{doc.get('passage', '')} {doc.get('question', '')}"
    return prompt, prompt + _drop_answer_text(doc.get("answer", {}))


def _nq_open_text(doc: dict[str, Any]) -> tuple[str, str]:
    prompt = f"Q: {doc.get('question', '')}?\nA: "
    return prompt, prompt + _stringify(_first(doc, "answer", "answers"))


def _triviaqa_text(doc: dict[str, Any]) -> tuple[str, str]:
    prompt = f"Question: {doc.get('question', '')}?\nAnswer: "
    answer = doc.get("answer", {})
    response = _stringify(answer.get("aliases") or answer.get("value") or answer)
    return prompt, prompt + response


def _meddialog_qsumm_text(doc: dict[str, Any]) -> tuple[str, str]:
    dialogue = _stringify(_first(doc, "src", "dialogue", "conversation", "input"))
    prompt = f"Medical dialogue:\n{dialogue}\n\nResponse: "
    return prompt, prompt + _stringify(_first(doc, "tgt", "response", "summary", "answer", "output"))


def _wmt16_en_de_text(doc: dict[str, Any]) -> tuple[str, str]:
    translation = doc.get("translation", {})
    prompt = f"English phrase: {translation.get('en', '')}\nGerman phrase:"
    return prompt, prompt + " " + translation.get("de", "")


def _wikitext_text(doc: dict[str, Any]) -> tuple[str, str]:
    return "", _stringify(_first(doc, "text", "page"))


def _cnn_dailymail_text(doc: dict[str, Any]) -> tuple[str, str]:
    prompt = f"Summarize the following article:\n\n{doc.get('article', '')}\n\nSummary: "
    return prompt, prompt + _stringify(doc.get("highlights", ""))


def _xsum_text(doc: dict[str, Any]) -> tuple[str, str]:
    prompt = f"Summarize the following document in one sentence:\n\n{doc.get('document', '')}\n\nSummary: "
    return prompt, prompt + _stringify(doc.get("summary", ""))


def _numinamath_text(doc: dict[str, Any]) -> tuple[str, str]:
    prompt = f"Problem:\n{doc['problem']}\n\nSolution:\n"
    return prompt, prompt + doc["solution"]


def _gsm8k_text(doc: dict[str, Any]) -> tuple[str, str]:
    prompt = f"Question: {doc.get('question', '')}\nAnswer: "
    return prompt, prompt + _stringify(doc.get("answer", ""))


def _babi_text(doc: dict[str, Any]) -> tuple[str, str]:
    prompt = f"Passage: {doc.get('passage', '')}Question: {doc.get('question', '')}\nAnswer: "
    return prompt, prompt + _stringify(doc.get("answer", ""))


def _squadv2_text(doc: dict[str, Any]) -> tuple[str, str]:
    prompt = (
        f"Title: {doc.get('title', '')}\n\n"
        f"Background: {doc.get('context', '')}\n\n"
        f"Question: {doc.get('question', '')}\n\n"
        "Answer: "
    )
    answers = doc.get("answers", {})
    answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
    target = answer_texts[0] if answer_texts else "unanswerable"
    return prompt, prompt + target


def _magicoder_text(doc: dict[str, Any]) -> tuple[str, str]:
    prompt = f"Problem:\n{doc['problem']}\n\nSolution:\n"
    return prompt, prompt + doc["solution"]


def _mbpp_text(doc: dict[str, Any]) -> tuple[str, str]:
    tests = doc.get("test_list", [])
    test_block = "\n".join(_stringify(test) for test in tests[:3])
    prompt = (
        f"You are an expert Python programmer, and here is your task: {doc.get('text', '')} "
        f"Your code should pass these tests:\n\n{test_block}\n[BEGIN]\n"
    )
    return prompt, prompt + _stringify(doc.get("code", ""))


def _humaneval_text(doc: dict[str, Any]) -> tuple[str, str]:
    return doc["prompt"], doc["prompt"] + doc["canonical_solution"]


# (tag, dataset_path, dataset_name, split, doc_to_text)
DATASET_2_TRAIN: list[TaskSpec] = [
    ("coqa", "EleutherAI/coqa", None, "train", _coqa_text),  # Conversational reading-comprehension QA with free-form answers.
    ("drop", "EleutherAI/drop", None, "train", _drop_text),  # Passage QA requiring discrete/numerical reasoning answers.
    ("nq_open", "google-research-datasets/nq_open", None, "train", _nq_open_text),  # Open-domain Natural Questions answer generation.
    ("triviaqa", "mandarjoshi/trivia_qa", "rc.nocontext", "train", _triviaqa_text),  # Trivia-style open QA with alias answer strings.
    ("meddialog_qsumm", "lighteval/med_dialog", "icliniq", "train", _meddialog_qsumm_text),  # Medical dialogue response/summarization generation.
    ("wmt16-en-de", "wmt/wmt16", "de-en", "train", _wmt16_en_de_text),  # English-to-German machine translation.
    (
        "wikitext",
        "EleutherAI/wikitext_document_level",
        "wikitext-2-raw-v1",
        "train",
        _wikitext_text,
    ),  # General-domain causal language modeling on Wikipedia text.
    ("cnn_dailymail", "abisee/cnn_dailymail", "3.0.0", "train", _cnn_dailymail_text),  # Multi-sentence news article summarization.
    ("xsum", "EdinburghNLP/xsum", None, "train", _xsum_text),  # Extreme one-sentence news summarization.
    ("gsm8k", "openai/gsm8k", "main", "train", _gsm8k_text),  # Grade-school math word-problem solution generation.
    ("babi", "Muennighoff/babi", None, "train", _babi_text),  # Synthetic story reasoning QA with short textual answers.
    ("squadv2", "lighteval/squad_v2", None, "train", _squadv2_text),  # Extractive or unanswerable passage QA.
    ("mbpp", "google-research-datasets/mbpp", "full", "train", _mbpp_text),  # Short Python program synthesis from natural-language tasks and tests.
    ("numinamath", "AI-MO/NuminaMath-TIR", None, "train", _numinamath_text),  # Competition-style mathematical solution generation.
    ("magicoder", "ise-uiuc/Magicoder-OSS-Instruct-75K", None, "train", _magicoder_text),  # Instruction-following code generation.
]

# (tag, dataset_path, dataset_name, split, doc_to_text)
DATASET_2_TEST: list[TaskSpec] = [
    ("coqa", "EleutherAI/coqa", None, "validation", _coqa_text),
    ("drop", "EleutherAI/drop", None, "validation", _drop_text),
    ("nq_open", "google-research-datasets/nq_open", None, "validation", _nq_open_text),
    ("triviaqa", "mandarjoshi/trivia_qa", "rc.nocontext", "validation", _triviaqa_text),
    ("meddialog_qsumm", "lighteval/med_dialog", "icliniq", "test", _meddialog_qsumm_text),
    ("wmt16-en-de", "wmt/wmt16", "de-en", "test", _wmt16_en_de_text),
    (
        "wikitext",
        "EleutherAI/wikitext_document_level",
        "wikitext-2-raw-v1",
        "test",
        _wikitext_text,
    ),
    ("cnn_dailymail", "abisee/cnn_dailymail", "3.0.0", "validation", _cnn_dailymail_text),
    ("xsum", "EdinburghNLP/xsum", None, "validation", _xsum_text),
    ("gsm8k", "openai/gsm8k", "main", "test", _gsm8k_text),
    ("babi", "Muennighoff/babi", None, "valid", _babi_text),
    ("squadv2", "lighteval/squad_v2", None, "validation", _squadv2_text),
    ("mbpp", "google-research-datasets/mbpp", "full", "test", _mbpp_text),
    ("math500", "HuggingFaceH4/MATH-500", "default", "test", _numinamath_text),
    ("humanevalplus", "evalplus/humanevalplus", None, "test", _humaneval_text),
 ]

DOC_TO_TEXT = {tag: formatter for tag, _, _, _, formatter in DATASET_2_TRAIN + DATASET_2_TEST}


def doc_to_text(task: str, doc: dict[str, Any]) -> tuple[str, str]:
    return DOC_TO_TEXT[task](doc)


def build_loader(
    dataset_path: str,
    dataset_name: str | None,
    split: str,
    doc_to_text_fn: Callable[[dict[str, Any]], tuple[str, str]],
    tokenizer,
    num_samples: int,
    batch_size: int,
    max_length: int,
    task: str | None = None,
) -> DataLoader:
    dataset = load_dataset(dataset_path, dataset_name, split=split, trust_remote_code=True)
    dataset = dataset.select(range(min(num_samples, len(dataset))))

    def join_prompt_and_target(prompt: str, target: str) -> tuple[list[int], list[int], list[int]]:
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) + len(target_ids) > max_length and target_ids:
            target_budget = min(len(target_ids), max_length - 1)
            prompt_budget = max_length - target_budget
            prompt_ids = prompt_ids[-prompt_budget:]
            target_ids = target_ids[:target_budget]
        input_ids = (prompt_ids + target_ids)[:max_length]
        labels = ([-100] * len(prompt_ids) + target_ids)[:max_length]
        attention_mask = [1] * len(input_ids)
        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len
            attention_mask += [0] * pad_len
        return input_ids, attention_mask, labels

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        source_docs = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
        docs = [
            expanded
            for doc in source_docs
            for expanded in expand_doc(task or "", doc)
        ]
        if not docs:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        prompts, texts = zip(*[doc_to_text_fn(doc) for doc in docs])
        rows = {"input_ids": [], "attention_mask": [], "labels": []}
        for prompt, text in zip(prompts, texts):
            target = text[len(prompt):]
            input_ids, attention_mask, labels = join_prompt_and_target(prompt, target)
            rows["input_ids"].append(input_ids)
            rows["attention_mask"].append(attention_mask)
            rows["labels"].append(labels)
        return rows

    dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return DataLoader(dataset, batch_size=batch_size)
