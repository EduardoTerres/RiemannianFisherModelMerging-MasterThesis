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

from src.dataset import dataset_1


class TinyTokenizer:
    pad_token_id = 0

    def __call__(self, texts, truncation=True, max_length=64, padding=False, add_special_tokens=True):
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
    return {
        "social_iqa": {"context": "Alex was hungry.", "question": "What next?", "answerA": "Eat", "answerB": "Sleep", "answerC": "Run", "label": "1"},
        "commonsense_qa": {"question": "Where do fish live?", "choices": {"text": ["sky", "water", "tree", "desk", "road"]}, "answerKey": "B"},
        "numinamath": {"problem": "1+1", "solution": "2"},
        "magicoder": {"problem": "print hi", "solution": "print('hi')"},
        "science_qa": {"question": "Which is hot?", "choices": ["ice", "fire"], "answer": 1},
        "math500": {"problem": "2+2", "solution": "4"},
        "humanevalplus": {"prompt": "def f():", "canonical_solution": "\n    return 1"},
    }[task]


def test_dataset_1_loaders_parse_all_specs(monkeypatch):
    def fake_load_dataset(path, name, split, trust_remote_code=True):
        task = next(tag for tag, p, n, s, _ in dataset_1.DATASET_1_TRAIN + dataset_1.DATASET_1_TEST if (p, n, s) == (path, name, split))
        return TinyDataset.from_list([sample_doc(task)])

    monkeypatch.setattr(dataset_1, "load_dataset", fake_load_dataset)
    tokenizer = TinyTokenizer()

    for task, path, name, split, formatter in dataset_1.DATASET_1_TRAIN + dataset_1.DATASET_1_TEST:
        loader = dataset_1.build_loader(path, name, split, formatter, tokenizer, num_samples=1, batch_size=1, max_length=256)
        batch = next(iter(loader))
        assert set(batch) == {"input_ids", "attention_mask", "labels"}
        assert batch["input_ids"].shape == batch["labels"].shape
        assert (batch["labels"] != -100).any(), task
