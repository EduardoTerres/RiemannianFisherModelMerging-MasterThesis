import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TinyDataset:
    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0]) if rows else []

    @classmethod
    def from_list(cls, rows):
        return cls(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return {
            key: torch.tensor(value) if isinstance(value, list) and all(isinstance(v, int) for v in value) else value
            for key, value in self.rows[idx].items()
        }

    def select(self, indices):
        return TinyDataset([self.rows[i] for i in indices])

    def map(self, fn, batched=False, remove_columns=None):
        if not batched:
            return TinyDataset([fn(row) for row in self.rows])
        batch = {key: [row[key] for row in self.rows] for key in self.column_names}
        mapped = fn(batch)
        n = len(next(iter(mapped.values()))) if mapped else 0
        return TinyDataset([{key: mapped[key][i] for key in mapped} for i in range(n)])

    def set_format(self, *args, **kwargs):
        return None


sys.modules.setdefault("datasets", types.SimpleNamespace(load_dataset=None))

from src.dataset import dataset_2


class TinyTokenizer:
    pad_token_id = 0

    def __call__(self, texts, truncation=True, max_length=96, padding=False, add_special_tokens=True):
        single = isinstance(texts, str)
        texts = [texts] if single else texts
        rows = []
        masks = []
        for text in texts:
            ids = [ord(ch) + 1 for ch in text][:max_length]
            mask = [1] * len(ids)
            if padding == "max_length":
                ids += [self.pad_token_id] * (max_length - len(ids))
                mask += [0] * (max_length - len(mask))
            rows.append(ids)
            masks.append(mask)
        out = {"input_ids": rows, "attention_mask": masks}
        return {"input_ids": rows[0], "attention_mask": masks[0]} if single else out


def sample_doc(task):
    docs = {
        "coqa": {"story": "A story.", "questions": {"input_text": ["Who?", "Why?"]}, "answers": {"input_text": ["Alex", "Because"]}},
        "drop": {"passage": "There were 3 cats.", "question": "How many cats?", "answer": {"number": "3", "spans": [], "date": {"day": "", "month": "", "year": ""}}},
        "nq_open": {"question": "Capital of France", "answer": ["Paris"]},
        "triviaqa": {"question": "Largest planet", "answer": {"aliases": ["Jupiter"], "value": "Jupiter"}},
        "jsonschema_bench_easy": {"json_schema": {"type": "object"}, "json_object": {"ok": True}},
        "meddialog_qsumm": {"dialogue": "Patient: cough", "response": "Likely cold"},
        "mimic_repsum": {"extractive_notes_summ": "FINDINGS chest clear     IMPRESSION no acute disease"},
        "wmt16-en-de": {"translation": {"en": "Hello", "de": "Hallo"}},
        "wikitext": {"page": "Some wiki text."},
        "cnn_dailymail": {"article": "Long news article.", "highlights": "Short news summary."},
        "xsum": {"document": "A news document with many details.", "summary": "A brief summary."},
        "gsm8k": {"question": "Mia has 2 apples and gets 3 more. How many?", "answer": "Mia has 2 + 3 = 5 apples. #### 5"},
        "babi": {"passage": "Mary went to the kitchen. ", "question": "Where is Mary?", "answer": "kitchen"},
        "squadv2": {"title": "France", "context": "Paris is the capital of France.", "question": "What is the capital?", "answers": {"text": ["Paris"]}},
        "numinamath": {"problem": "1+1", "solution": "2"},
        "magicoder": {"problem": "print hi", "solution": "print('hi')"},
        "mbpp": {"text": "Write a function that returns one.", "test_list": ["assert f() == 1"], "code": "def f():\n    return 1"},
        "math500": {"problem": "2+2", "solution": "4"},
        "humanevalplus": {"prompt": "def f():", "canonical_solution": "\n    return 1"},
    }
    return docs[task]


def test_dataset_2_loaders_parse_all_specs(monkeypatch):
    specs = dataset_2.DATASET_2_TRAIN + dataset_2.DATASET_2_TEST

    def fake_load_dataset(path, name, split, trust_remote_code=True):
        task = next(tag for tag, p, n, s, _ in specs if (p, n, s) == (path, name, split))
        return TinyDataset.from_list([sample_doc(task)])

    monkeypatch.setattr(dataset_2, "load_dataset", fake_load_dataset)
    tokenizer = TinyTokenizer()

    for task, path, name, split, formatter in specs:
        loader = dataset_2.build_loader(path, name, split, formatter, tokenizer, num_samples=1, batch_size=1, max_length=512, task=task)
        batch = next(iter(loader))
        assert set(batch) == {"input_ids", "attention_mask", "labels"}
        assert batch["input_ids"].shape == batch["labels"].shape
        assert (batch["labels"] != -100).any(), task
