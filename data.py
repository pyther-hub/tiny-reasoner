from functools import partial

import torch
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset, Dataset


def load_gsm8k(val_size=256, seed=42):
    dataset = load_dataset("openai/gsm8k", "main")
    train_val = dataset["train"].train_test_split(test_size=val_size, seed=seed)
    return train_val["train"], train_val["test"], dataset["test"]


def tiny_dataset(n=8):
    """Fake GSM8K-shaped rows for smoke tests, no download."""
    rows = [
        {"question": f"Tom has {i} apples and buys {i + 2} more. How many apples?",
         "answer": f"{i} + {i + 2} = {2 * i + 2}\n#### {2 * i + 2}"}
        for i in range(n)
    ]
    return Dataset.from_list(rows)


def format_prompt(question):
    return f"Question: {question}\nAnswer:"


def collate_fn(batch, tokenizer, max_length, max_input_length):
    input_ids_list = []
    labels_list = []

    for ex in batch:
        prompt = format_prompt(ex["question"])
        answer = f"{ex['answer']}{tokenizer.eos_token}"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][-max_input_length:]
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"][:max_length - len(prompt_ids)]
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        input_ids_list.append(input_ids)
        labels_list.append(labels)

    batch_max_len = max(len(ids) for ids in input_ids_list)
    padded_input_ids, padded_labels, attention_masks = [], [], []

    for input_ids, labels in zip(input_ids_list, labels_list):
        pad_len = batch_max_len - len(input_ids)
        padded_input_ids.append(input_ids + [tokenizer.pad_token_id] * pad_len)
        padded_labels.append(labels + [-100] * pad_len)
        attention_masks.append([1] * len(input_ids) + [0] * pad_len)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
    }


def make_loader(dataset, tokenizer, batch_size, max_length=384, max_input_length=128, shuffle=True,
                distributed=False, num_workers=0):
    """distributed=True: each rank gets its own shard. Call loader.sampler.set_epoch(epoch) every epoch."""
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        # partial, not lambda: must be picklable for num_workers > 0 and spawned processes
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            max_length=max_length,
            max_input_length=max_input_length,
        ),
    )
