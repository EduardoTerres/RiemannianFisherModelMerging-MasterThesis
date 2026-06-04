import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault("datasets", types.SimpleNamespace(load_dataset=None))

from src.dataset import dataset_2


SEED = 42


class CharTokenizer:
    pad_token_id = 0
    bos_token_id = 1

    def __call__(self, text, add_special_tokens=True):
        ids = [ord(ch) + 2 for ch in text]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        return {"input_ids": ids}

    def decode(self, ids):
        return "".join(
            chr(token - 2)
            for token in ids
            if token not in {self.pad_token_id, self.bos_token_id}
        )


def mask_prompt(
    text: str,
    prompt: str,
    tokenizer: CharTokenizer,
    max_length: int | None = None,
) -> tuple[str, str]:
    target = text[len(prompt):]
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    if max_length is not None and len(prompt_ids) + len(target_ids) > max_length and target_ids:
        target_budget = min(len(target_ids), max_length - 1)
        prompt_budget = max_length - target_budget
        prompt_ids = prompt_ids[-prompt_budget:]
        target_ids = target_ids[:target_budget]
    ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    if max_length is not None:
        ids = ids[:max_length]
        labels = labels[:max_length]
    prompt_masked = tokenizer.decode([token for token, label in zip(ids, labels) if label == -100])
    label_text = tokenizer.decode([label for label in labels if label != -100])
    return prompt_masked, label_text


def decode_prompt_tokens(prompt: str, tokenizer: CharTokenizer) -> str:
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    return tokenizer.decode(prompt_ids)


EXAMPLE_CASES = {
    "coqa": [
        (
            {
                "story": "Alex put the blue cup on the table.",
                "questions": {"input_text": ["Who put the cup down?", "Where is the cup?"]},
                "answers": {"input_text": ["Alex", "on the table"]},
            },
            "Alex put the blue cup on the table.\n\nQ: Who put the cup down?\n\nA: Alex\n\nQ: Where is the cup?\n\nA:",
            "on the table",
        ),
        (
            {
                "story": "Sam opened the red door.",
                "questions": {"input_text": ["Who opened the door?", "What color was it?"]},
                "answers": {"input_text": ["Sam", "red"]},
            },
            "Sam opened the red door.\n\nQ: Who opened the door?\n\nA: Sam\n\nQ: What color was it?\n\nA:",
            "red",
        ),
    ],
    "drop": [
        (
            {
                "passage": "The Eagles won 6 games, lost 2 games, and tied 1 game.",
                "question": "How many games did the Eagles not win?",
                "answer": {"number": "3", "spans": [], "date": {"day": "", "month": "", "year": ""}},
            },
            "The Eagles won 6 games, lost 2 games, and tied 1 game. How many games did the Eagles not win?",
            "3",
        ),
        (
            {
                "passage": "The team scored 4 goals in March and 5 goals in April.",
                "question": "How many goals did the team score in total?",
                "answer": {"number": "9", "spans": [], "date": {"day": "", "month": "", "year": ""}},
            },
            "The team scored 4 goals in March and 5 goals in April. How many goals did the team score in total?",
            "9",
        ),
    ],
    "nq_open": [
        (
            {"question": "What is the capital of France?", "answer": ["Paris"]},
            "Q: What is the capital of France??\nA: ",
            "Paris",
        ),
        (
            {"question": "Who wrote Hamlet", "answer": ["William Shakespeare"]},
            "Q: Who wrote Hamlet?\nA: ",
            "William Shakespeare",
        ),
    ],
    "triviaqa": [
        (
            {"question": "Largest planet", "answer": {"aliases": ["Jupiter"], "value": "Jupiter"}},
            "Question: Largest planet?\nAnswer: ",
            "Jupiter",
        ),
        (
            {"question": "Smallest prime number", "answer": {"aliases": ["2"], "value": "2"}},
            "Question: Smallest prime number?\nAnswer: ",
            "2",
        ),
    ],
    "meddialog_qsumm": [
        (
            {"src": "Patient: I have a cough.", "tgt": "Likely viral infection."},
            "Medical dialogue:\nPatient: I have a cough.\n\nResponse: ",
            "Likely viral infection.",
        ),
        (
            {"src": "Patient: My throat hurts.", "tgt": "Sore throat complaint."},
            "Medical dialogue:\nPatient: My throat hurts.\n\nResponse: ",
            "Sore throat complaint.",
        ),
    ],
    "wmt16-en-de": [
        (
            {"translation": {"en": "Hello world", "de": "Hallo Welt"}},
            "English phrase: Hello world\nGerman phrase:",
            " Hallo Welt",
        ),
        (
            {"translation": {"en": "Good morning", "de": "Guten Morgen"}},
            "English phrase: Good morning\nGerman phrase:",
            " Guten Morgen",
        ),
    ],
    "wikitext": [
        ({"page": "Some wiki text."}, "", "Some wiki text."),
        ({"text": "Another wiki paragraph."}, "", "Another wiki paragraph."),
    ],
    "cnn_dailymail": [
        (
            {"article": "A long article.", "highlights": "A short summary."},
            "Summarize the following article:\n\nA long article.\n\nSummary: ",
            "A short summary.",
        ),
        (
            {"article": "Another article.", "highlights": "Another summary."},
            "Summarize the following article:\n\nAnother article.\n\nSummary: ",
            "Another summary.",
        ),
    ],
    "xsum": [
        (
            {"document": "A document with many details.", "summary": "A one sentence summary."},
            "Summarize the following document in one sentence:\n\nA document with many details.\n\nSummary: ",
            "A one sentence summary.",
        ),
        (
            {"document": "Another document.", "summary": "Another one sentence summary."},
            "Summarize the following document in one sentence:\n\nAnother document.\n\nSummary: ",
            "Another one sentence summary.",
        ),
    ],
    "gsm8k": [
        (
            {"question": "Mia has 2 apples and gets 3 more. How many?", "answer": "5"},
            "Question: Mia has 2 apples and gets 3 more. How many?\nAnswer: ",
            "5",
        ),
        (
            {"question": "There are 7 birds and 2 fly away. How many remain?", "answer": "5"},
            "Question: There are 7 birds and 2 fly away. How many remain?\nAnswer: ",
            "5",
        ),
    ],
    "babi": [
        (
            {"passage": "Mary went to the kitchen. ", "question": "Where is Mary?", "answer": "kitchen"},
            "Passage: Mary went to the kitchen. Question: Where is Mary?\nAnswer: ",
            "kitchen",
        ),
        (
            {"passage": "John travelled to the office. ", "question": "Where is John?", "answer": "office"},
            "Passage: John travelled to the office. Question: Where is John?\nAnswer: ",
            "office",
        ),
    ],
    "squadv2": [
        (
            {
                "title": "France",
                "context": "Paris is the capital of France.",
                "question": "What is the capital?",
                "answers": {"text": ["Paris"]},
            },
            "Title: France\n\nBackground: Paris is the capital of France.\n\nQuestion: What is the capital?\n\nAnswer: ",
            "Paris",
        ),
        (
            {
                "title": "Ocean",
                "context": "The Pacific is the largest ocean.",
                "question": "Which ocean is largest?",
                "answers": {"text": ["Pacific"]},
            },
            "Title: Ocean\n\nBackground: The Pacific is the largest ocean.\n\nQuestion: Which ocean is largest?\n\nAnswer: ",
            "Pacific",
        ),
    ],
    "mbpp": [
        (
            {
                "text": "Write a function that returns one.",
                "test_list": ["assert f() == 1"],
                "code": "def f():\n    return 1",
            },
            "You are an expert Python programmer, and here is your task: Write a function that returns one. Your code should pass these tests:\n\nassert f() == 1\n[BEGIN]\n",
            "def f():\n    return 1",
        ),
        (
            {
                "text": "Write a function that doubles a number.",
                "test_list": ["assert double(2) == 4", "assert double(3) == 6"],
                "code": "def double(x):\n    return 2 * x",
            },
            "You are an expert Python programmer, and here is your task: Write a function that doubles a number. Your code should pass these tests:\n\nassert double(2) == 4\nassert double(3) == 6\n[BEGIN]\n",
            "def double(x):\n    return 2 * x",
        ),
    ],
    "numinamath": [
        ({"problem": "Compute 1+1.", "solution": "2"}, "Problem:\nCompute 1+1.\n\nSolution:\n", "2"),
        ({"problem": "Compute 3+4.", "solution": "7"}, "Problem:\nCompute 3+4.\n\nSolution:\n", "7"),
    ],
    "magicoder": [
        (
            {"problem": "Write Python code that prints hi.", "solution": "print('hi')"},
            "Problem:\nWrite Python code that prints hi.\n\nSolution:\n",
            "print('hi')",
        ),
        (
            {"problem": "Write Python code that returns one.", "solution": "return 1"},
            "Problem:\nWrite Python code that returns one.\n\nSolution:\n",
            "return 1",
        ),
    ],
    "math500": [
        ({"problem": "Compute 2+2.", "solution": "4"}, "Problem:\nCompute 2+2.\n\nSolution:\n", "4"),
        ({"problem": "Compute 5-2.", "solution": "3"}, "Problem:\nCompute 5-2.\n\nSolution:\n", "3"),
    ],
    "humanevalplus": [
        ({"prompt": "def f():", "canonical_solution": "\n    return 1"}, "def f():", "\n    return 1"),
        ({"prompt": "def g(x):", "canonical_solution": "\n    return x"}, "def g(x):", "\n    return x"),
    ],
}

EXAMPLES = [
    pytest.param(task, doc, expected_prompt, expected_labels, id=f"{task}-{i}")
    for task, cases in EXAMPLE_CASES.items()
    for i, (doc, expected_prompt, expected_labels) in enumerate(cases)
]


def test_all_dataset_2_tasks_are_covered():
    expected_tasks = {task for task, *_ in dataset_2.DATASET_2_TRAIN + dataset_2.DATASET_2_TEST}
    expected_tasks.update({"numinamath", "magicoder"})
    tested_tasks = set(EXAMPLE_CASES)
    assert tested_tasks == expected_tasks


def test_dataset_2_long_meddialog_prompt_keeps_labels_after_truncation():
    doc = {"src": " ".join(["symptom"] * 80), "tgt": "short diagnosis"}
    prompt, text = dataset_2.doc_to_text("meddialog_qsumm", doc)
    prompt_masked, labels = mask_prompt(text, prompt, CharTokenizer(), max_length=40)

    assert prompt_masked
    assert labels == "short diagnosis"


@pytest.mark.parametrize(("task", "doc", "expected_prompt", "expected_labels"), EXAMPLES)
def test_dataset_2_prompt_tokens_are_correct(task, doc, expected_prompt, expected_labels):
    formatter = {
        "numinamath": dataset_2._numinamath_text,
        "magicoder": dataset_2._magicoder_text,
    }.get(task, dataset_2.doc_to_text)
    prompt, _ = formatter(task, doc) if formatter is dataset_2.doc_to_text else formatter(doc)

    assert prompt == expected_prompt
    assert decode_prompt_tokens(prompt, CharTokenizer()) == expected_prompt


@pytest.mark.parametrize(("task", "doc", "expected_prompt", "expected_labels"), EXAMPLES)
def test_dataset_2_prompt_mask_and_labels_are_correct(task, doc, expected_prompt, expected_labels):
    assert SEED == 42

    formatter = {
        "numinamath": dataset_2._numinamath_text,
        "magicoder": dataset_2._magicoder_text,
    }.get(task, dataset_2.doc_to_text)
    prompt, text = formatter(task, doc) if formatter is dataset_2.doc_to_text else formatter(doc)
    prompt_masked, labels = mask_prompt(text, prompt, CharTokenizer())

    assert prompt == expected_prompt
    assert text == expected_prompt + expected_labels
    assert prompt_masked == expected_prompt
    assert labels == expected_labels
