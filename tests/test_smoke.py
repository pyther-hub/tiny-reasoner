"""Fast CPU checks with a tiny random model. Run after every edit:
    .venv/bin/python -m pytest -q
"""
import pytest
import torch
from torch.optim import AdamW

from data import tiny_dataset, collate_fn, make_loader
from model import tiny_model, load_tokenizer
from train import forward_loss, train_one_epoch, eval_val_loss, check_grads
from eval import extract_answer, evaluate_gsm8k

DEVICE = torch.device("cpu")
TOKENIZER_NAME = "openai-community/gpt2"  # small and cached; real runs use Qwen


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(TOKENIZER_NAME)


@pytest.fixture
def batch(tokenizer):
    ds = tiny_dataset(4)
    return collate_fn([ds[i] for i in range(4)], tokenizer, max_length=64, max_input_length=32)


def test_collate_shapes(batch, tokenizer):
    B, T = batch["input_ids"].shape
    assert B == 4 and T <= 64
    assert batch["labels"].shape == (B, T) == batch["attention_mask"].shape
    # padding must be masked out of both attention and loss
    pad = batch["attention_mask"] == 0
    assert (batch["labels"][pad] == -100).all()
    # every row has some answer tokens to learn from
    assert ((batch["labels"] != -100).sum(dim=1) > 0).all()


def test_forward_shapes(batch, tokenizer):
    torch.manual_seed(0)
    model = tiny_model(len(tokenizer))
    loss, logits = forward_loss(model, batch, DEVICE, debug=True)
    assert logits.shape == (*batch["input_ids"].shape, len(tokenizer))
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_lora_grads(batch, tokenizer):
    torch.manual_seed(0)
    model = tiny_model(len(tokenizer), lora=True)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
    loss, _ = forward_loss(model, batch, DEVICE)
    loss.backward()
    check_grads(model)


def test_overfit_one_batch(batch, tokenizer):
    torch.manual_seed(0)
    model = tiny_model(len(tokenizer), lora=False)
    optimizer = AdamW(model.parameters(), lr=3e-3)
    model.train()
    first = None
    for _ in range(60):
        loss, _ = forward_loss(model, batch, DEVICE)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.3 * first, f"did not overfit: {first:.3f} -> {loss.item():.3f}"


def test_debug_pipeline(tokenizer):
    torch.manual_seed(0)
    model = tiny_model(len(tokenizer))
    ds = tiny_dataset(8)
    loader = make_loader(ds, tokenizer, batch_size=2, max_length=64, max_input_length=32)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    losses = train_one_epoch(model, loader, optimizer, DEVICE, debug=True)
    assert len(losses) == 3
    assert eval_val_loss(model, loader, DEVICE, debug=True) > 0
    acc, preds = evaluate_gsm8k(model, tokenizer, ds, DEVICE, batch_size=2, debug=True)
    assert 0.0 <= acc <= 1.0 and len(preds) == 2


def test_extract_answer():
    assert extract_answer("so 3 + 4 = 7\n#### 7") == "7"
    assert extract_answer("#### 1,234") == "1234"
    assert extract_answer("#### -2.5") == "-2.5"
    assert extract_answer("no answer here") is None
