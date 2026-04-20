"""Dataset set 1: 5 tasks for evaluating the 5 adapters fine-tuned on Llama-3.1-8B."""

def _siqa_text(doc):
    choices = [doc["answerA"], doc["answerB"], doc["answerC"]]
    correct = choices[int(doc["label"]) - 1]
    return f"Q: {doc['context']} {doc['question']}\nA: {correct}"

def _csqa_text(doc):
    letters = ["A", "B", "C", "D", "E"]
    choices_text = "\n".join(
        f"{l}. {t}" for l, t in zip(letters, doc["choices"]["text"])
    )
    return f"Question: {doc['question'].strip()}\n{choices_text}\nAnswer: {doc['answerKey']}"

def _minerva_text(doc):
    return f"Problem:\n{doc['problem']}\n\nSolution:\n{doc['solution']}"

def _humaneval_text(doc):
    return doc["prompt"] + doc["canonical_solution"]

def _scienceqa_text(example):
    """Format defined in eval_scienceqa.py in eval-harness."""
    _INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

    question = example["question"]
    options = example["choices"]  # List[str]

    choice_lines = []
    for i, opt in enumerate(options):
        label = _INDEX_TO_LETTER.get(i, chr(ord("A") + i))
        choice_lines.append(f"{label}. {opt}")
    choices_str = "\n".join(choice_lines)

    prompt = (
        f"Question: {question}\n"
        f"Choices:\n{choices_str}\n\n"
        f"Answer:"
    )
    return prompt

DATASET_1 = [
    # (tag, dataset_path, dataset_name, split, doc_to_text, adapter_path)
    ("social_iqa",      "allenai/social_i_qa",    None,      "train", _siqa_text),
    ("commonsense_qa",  "tau/commonsense_qa",      None,      "train", _csqa_text),
    ("numinamath", "HuggingFaceH4/MATH-500",  "default", "test",  _minerva_text),
    ("humanevalplus",   "evalplus/humanevalplus",  None,      "test",  _humaneval_text),
    ("science_qa",      "derek-thomas/ScienceQA",     None,    "train", _scienceqa_text),
]
