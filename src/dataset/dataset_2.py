"""Dataset set 2: extra fine-tuning tasks plus math and code tasks from set 1."""

from __future__ import annotations

import json
from typing import Any, Callable


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


def _coqa_text(doc: dict[str, Any]) -> tuple[str, str]:
    qs = doc.get("questions", [])
    answers = doc.get("answers", {})
    ans = answers.get("input_text", answers) if isinstance(answers, dict) else answers
    turns = "\n".join(f"Q: {_stringify(q)}\nA: {_stringify(a)}" for q, a in zip(qs, ans))
    prompt = f"Passage:\n{doc.get('story', '')}\n\nAnswer the questions.\n"
    return prompt, prompt + turns


def _drop_text(doc: dict[str, Any]) -> tuple[str, str]:
    prompt = (
        f"Passage:\n{doc.get('passage', '')}\n\n"
        f"Question: {doc.get('question', '')}\nAnswer: "
    )
    return prompt, prompt + _stringify(_first(doc, "answer", "validated_answers", "answers"))


def _qasper_freeform_text(doc: dict[str, Any]) -> tuple[str, str]:
    abstract = _stringify(_first(doc, "abstract", "full_text"))
    prompt = (
        f"Paper:\n{doc.get('title', '')}\n{abstract}\n\n"
        f"Question: {doc.get('question', '')}\nAnswer: "
    )
    return prompt, prompt + _stringify(_first(doc, "answers", "answer"))


def _nq_open_text(doc: dict[str, Any]) -> tuple[str, str]:
    prompt = f"Question: {doc.get('question', '')}\nAnswer: "
    return prompt, prompt + _stringify(_first(doc, "answer", "answers"))


def _triviaqa_text(doc: dict[str, Any]) -> tuple[str, str]:
    prompt = f"Question: {doc.get('question', '')}\nAnswer: "
    answer = doc.get("answer", {})
    response = answer.get("value") or _stringify(answer.get("aliases", answer))
    return prompt, prompt + response


def _jsonschema_bench_easy_text(doc: dict[str, Any]) -> tuple[str, str]:
    schema = _first(doc, "schema", "json_schema", "output_schema")
    prompt = f"Generate JSON matching this schema:\n{_stringify(schema)}\n\nJSON: "
    return prompt, prompt + _stringify(_first(doc, "instance", "output", "answer", "json"))


def _meddialog_qsumm_text(doc: dict[str, Any]) -> tuple[str, str]:
    dialogue = _stringify(_first(doc, "dialogue", "conversation", "input"))
    prompt = f"Medical dialogue:\n{dialogue}\n\nResponse: "
    return prompt, prompt + _stringify(_first(doc, "response", "summary", "answer", "output"))


def _mimic_repsum_text(doc: dict[str, Any]) -> tuple[str, str]:
    course = _stringify(_first(doc, "text", "input", "hospital_course"))
    prompt = f"Hospital course:\n{course}\n\nSummary: "
    return prompt, prompt + _stringify(_first(doc, "summary", "target", "output"))


def _wmt16_en_de_text(doc: dict[str, Any]) -> tuple[str, str]:
    translation = doc.get("translation", {})
    prompt = f"Translate English to German:\n{translation.get('en', '')}\n\nGerman: "
    return prompt, prompt + translation.get("de", "")


def _wikitext_text(doc: dict[str, Any]) -> tuple[str, str]:
    return "", _stringify(_first(doc, "text", "page"))


def _numinamath_text(doc: dict[str, Any]) -> tuple[str, str]:
    prompt = f"Problem:\n{doc['problem']}\n\nSolution:\n"
    return prompt, prompt + doc["solution"]


def _magicoder_text(doc: dict[str, Any]) -> tuple[str, str]:
    prompt = f"Problem:\n{doc['problem']}\n\nSolution:\n"
    return prompt, prompt + doc["solution"]


DATASET_2_TRAIN: list[TaskSpec] = [
    ("coqa", "EleutherAI/coqa", None, "train", _coqa_text),
    ("drop", "EleutherAI/drop", None, "train", _drop_text),
    ("qasper_freeform", "allenai/qasper", None, "train", _qasper_freeform_text),
    ("nq_open", "google-research-datasets/nq_open", None, "train", _nq_open_text),
    ("triviaqa", "mandarjoshi/trivia_qa", "rc.nocontext", "train", _triviaqa_text),
    (
        "jsonschema_bench_easy",
        "epfl-dlab/JSONSchemaBench",
        "Github_easy",
        "train",
        _jsonschema_bench_easy_text,
    ),
    ("meddialog_qsumm", "lighteval/med_dialog", "icliniq", "train", _meddialog_qsumm_text),
    (
        "mimic_repsum",
        "dmacres/mimiciii-hospitalcourse-meta",
        None,
        "train",
        _mimic_repsum_text,
    ),
    ("wmt16-en-de", "wmt/wmt16", "de-en", "train", _wmt16_en_de_text),
    (
        "wikitext",
        "EleutherAI/wikitext_document_level",
        "wikitext-2-raw-v1",
        "train",
        _wikitext_text,
    ),
    ("numinamath", "AI-MO/NuminaMath-TIR", None, "train", _numinamath_text),
    ("magicoder", "ise-uiuc/Magicoder-OSS-Instruct-75K", None, "train", _magicoder_text),
]


DOC_TO_TEXT = {tag: formatter for tag, _, _, _, formatter in DATASET_2_TRAIN}


def doc_to_text(task: str, doc: dict[str, Any]) -> tuple[str, str]:
    return DOC_TO_TEXT[task](doc)
