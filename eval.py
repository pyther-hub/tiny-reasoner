import re

import torch
from tqdm.auto import tqdm

from data import format_prompt


@torch.no_grad()
def generate_seq(model, tokenizer, list_of_questions, device, max_seq_len=512):
    """Greedy decoding with left padding. Returns only the generated text."""
    input_seq_ids = [
        tokenizer(format_prompt(q), add_special_tokens=False)["input_ids"] for q in list_of_questions
    ]

    max_input_seq = max(len(x) for x in input_seq_ids)
    padded_input_ids = []
    attention_masks = []
    for input_ids in input_seq_ids:
        left_side_pad_len = max_input_seq - len(input_ids)
        padded_input_ids.append([tokenizer.pad_token_id] * left_side_pad_len + input_ids)
        attention_masks.append([0] * left_side_pad_len + [1] * len(input_ids))

    x_in = torch.tensor(padded_input_ids, dtype=torch.long, device=device)
    attn_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)

    model.eval()
    finished = torch.zeros(x_in.size(0), dtype=torch.bool, device=device)
    for _ in range(max_seq_len):
        logits = model(input_ids=x_in, attention_mask=attn_mask).logits[:, -1, :]
        preds_token_id = logits.argmax(dim=-1)
        preds_token_id = torch.where(finished, tokenizer.eos_token_id, preds_token_id)

        x_in = torch.cat([x_in, preds_token_id.unsqueeze(1)], dim=1)
        attn_mask = torch.cat([attn_mask, (~finished).long().unsqueeze(1)], dim=1)
        finished = finished | preds_token_id.eq(tokenizer.eos_token_id)
        if finished.all():
            break
    return [tokenizer.decode(seq[max_input_seq:], skip_special_tokens=True) for seq in x_in]


def extract_answer(text):
    matches = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    return matches[-1].replace(",", "") if matches else None


def evaluate_gsm8k(model, tokenizer, test_dataset, device, batch_size=8, max_new_tokens=256, debug=False):
    """Exact-match accuracy on the #### answer. debug=True: one batch, 8 new tokens."""
    if debug:
        test_dataset = test_dataset.select(range(min(batch_size, len(test_dataset))))
        max_new_tokens = 8
    model.eval()
    correct = 0
    total = 0
    predictions = []
    for start in tqdm(range(0, len(test_dataset), batch_size), disable=debug):
        batch = test_dataset.select(range(start, min(start + batch_size, len(test_dataset))))
        questions = batch["question"]
        references = [extract_answer(a) for a in batch["answer"]]
        generated = generate_seq(model, tokenizer, questions, device, max_seq_len=max_new_tokens)
        for question, prediction, reference in zip(questions, generated, references):
            predicted_answer = extract_answer(prediction)
            is_correct = predicted_answer is not None and reference is not None and predicted_answer == reference
            correct += int(is_correct)
            total += 1
            predictions.append({
                "question": question,
                "prediction": prediction,
                "predicted_answer": predicted_answer,
                "reference": reference,
                "correct": is_correct,
            })
    accuracy = correct / total if total else 0.0
    print(f"Test accuracy: {accuracy:.2%} ({correct}/{total})")
    return accuracy, predictions


def show_samples(model, tokenizer, dataset, device, n=5, max_new_tokens=64):
    questions = dataset[:n]["question"]
    outputs = generate_seq(model, tokenizer, questions, device, max_seq_len=max_new_tokens)
    for i, (question, output, answer) in enumerate(zip(questions, outputs, dataset[:n]["answer"]), 1):
        print(f"\nExample {i}\nQuestion: {question}\nGenerated: {output}\nExpected: {answer}")
        print("-" * 80)
