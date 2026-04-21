"""Dataset set 1: 5 tasks for evaluating the 5 adapters fine-tuned on Llama-3.1-8B."""

from collections.abc import Callable

from datasets import load_dataset
from torch.utils.data import DataLoader


def build_loader(
    dataset_path: str,
    dataset_name: str | None,
    split: str,
    doc_to_text: Callable[[dict], tuple[str, str]],
    tokenizer,
    num_samples: int,
    batch_size: int,
    max_length: int,
) -> DataLoader:
    dataset = load_dataset(dataset_path, dataset_name, split=split, trust_remote_code=True)
    dataset = dataset.select(range(min(num_samples, len(dataset))))

    def tokenize(batch):
        n = len(batch[next(iter(batch))])
        docs = [{k: batch[k][i] for k in batch} for i in range(n)]
        prompts, texts = zip(*[doc_to_text(d) for d in docs])
        enc = tokenizer(list(texts), truncation=True, max_length=max_length, padding="max_length")
        prompt_lens = [
            len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts
        ]
        labels = [
            [-100 if (j < plen or attn == 0) else ids[j]
            for j, attn in enumerate(enc["attention_mask"][i])]
            for i, (ids, plen) in enumerate(zip(enc["input_ids"], prompt_lens))
        ]
        enc["labels"] = labels
        return enc

    dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return DataLoader(dataset, batch_size=batch_size)

def _siqa_text(doc):
    prompt = f"Q: {doc['context']} {doc['question']}\nA: "
    choices = [doc["answerA"], doc["answerB"], doc["answerC"]]
    response = choices[int(doc["label"]) - 1]
    return prompt, prompt + response

def _csqa_text(doc):
    letters = ["A", "B", "C", "D", "E"]
    choices_text = "\n".join(
        f"{letter}. {t}" for letter, t in zip(letters, doc["choices"]["text"])
    )
    prompt = f"Question: {doc['question'].strip()}\n{choices_text}\nAnswer: "
    return prompt, prompt + doc["answerKey"]

def _minerva_math500_text(doc):
    prompt = f"Problem:\n{doc['problem']}\n\nSolution:\n"
    return prompt, prompt + doc["solution"]

def _humaneval_text(doc):
    return doc["prompt"], doc["prompt"] + doc["canonical_solution"]

def _numinamath_text(doc):
    prompt = f"Problem:\n{doc['problem']}\n\nSolution:\n"
    return prompt, prompt + doc["solution"]

def _magicoder_text(doc):
    prompt = f"Problem:\n{doc['problem']}\n\nSolution:\n"
    return prompt, prompt + doc["solution"]

def _scienceqa_text(example):
    """Format defined in eval_scienceqa.py in eval-harness."""
    _INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

    question = example["question"]
    options = example["choices"]
    answer_letter = _INDEX_TO_LETTER.get(int(example["answer"]), "A")

    choice_lines = []
    for i, opt in enumerate(options):
        label = _INDEX_TO_LETTER.get(i, chr(ord("A") + i))
        choice_lines.append(f"{label}. {opt}")
    choices_str = "\n".join(choice_lines)

    prompt = f"Question: {question}\nChoices:\n{choices_str}\n\nAnswer: "
    return prompt, prompt + answer_letter

# (tag, dataset_path, dataset_name, split, doc_to_text)
DATASET_1_TRAIN = [
    ("social_iqa",     "allenai/social_i_qa",   None,      "train", _siqa_text),
    ("commonsense_qa", "tau/commonsense_qa",     None,      "train", _csqa_text),
    ("numinamath",     "AI-MO/NuminaMath-TIR", None,      "train",  _numinamath_text),
    ("magicoder",  "ise-uiuc/Magicoder-OSS-Instruct-75K", None,      "train",  _magicoder_text),
    ("science_qa",     "derek-thomas/ScienceQA", None,      "train", _scienceqa_text)
]
DATASET_1_TEST = [
    ("social_iqa",     "allenai/social_i_qa",   None,      "validation", _siqa_text),
    ("commonsense_qa", "tau/commonsense_qa",     None,      "validation", _csqa_text),
    ("math500",     "HuggingFaceH4/MATH-500", "default", "test",  _minerva_math500_text),
    ("humanevalplus",  "evalplus/humanevalplus", None,      "test",  _humaneval_text),
    ("science_qa",     "derek-thomas/ScienceQA", None,      "test", _scienceqa_text),
]
