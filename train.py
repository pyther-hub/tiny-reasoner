import gc

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

DEBUG_STEPS = 3


def calc_loss(logits, labels):
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )


def print_shapes(**tensors):
    for name, t in tensors.items():
        print(f"  {name:<14} {tuple(t.shape)}  {t.dtype}  {t.device}")


def forward_loss(model, batch, device, debug=False):
    """Shifted next-token loss. Returns (loss, logits)."""
    x_in = batch["input_ids"].to(device)
    y_label = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    logits = model(input_ids=x_in, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = y_label[:, 1:].contiguous()
    loss = calc_loss(shift_logits, shift_labels)

    if debug:
        print_shapes(input_ids=x_in, attention_mask=attention_mask, labels=y_label,
                     logits=logits, shift_logits=shift_logits, shift_labels=shift_labels)
        print(f"  loss           {loss.item():.4f}")
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite loss: {loss.item()}")
    return loss, logits


def train_one_epoch(model, train_loader, optimizer, device, accum_steps=4, debug=False):
    """debug=True: run DEBUG_STEPS batches, print tensor shapes on the first, check grads."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    num_batches = min(len(train_loader), DEBUG_STEPS) if debug else len(train_loader)
    if debug:
        accum_steps = 1
    losses = []

    with tqdm(enumerate(train_loader), total=num_batches, disable=debug) as pbar:
        for batch_idx, batch in pbar:
            if batch_idx >= num_batches:
                break
            first = debug and batch_idx == 0
            if first:
                print("[debug] first batch:")
            loss, _ = forward_loss(model, batch, device, debug=first)

            (loss / accum_steps).backward()
            if first:
                check_grads(model)

            is_update_step = (batch_idx + 1) % accum_steps == 0
            is_last_batch = (batch_idx + 1) == num_batches
            if is_update_step or is_last_batch:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item())
            if debug:
                print(f"[debug] step {batch_idx + 1}/{num_batches} loss={loss.item():.4f}")
            pbar.set_postfix({"train_loss": loss.item()})
    return losses


@torch.no_grad()
def eval_val_loss(model, val_loader, device, debug=False):
    model.eval()
    num_batches = min(len(val_loader), DEBUG_STEPS) if debug else len(val_loader)
    total_loss = 0.0
    with tqdm(enumerate(val_loader), total=num_batches, disable=debug) as pbar:
        for batch_idx, batch in pbar:
            if batch_idx >= num_batches:
                break
            loss, _ = forward_loss(model, batch, device)
            total_loss += loss.item()
            pbar.set_postfix({"val_loss": total_loss / (batch_idx + 1)})
    return total_loss / num_batches


def check_grads(model):
    """Every trainable param must get a finite, non-None gradient."""
    missing, bad = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            missing.append(name)
        elif not torch.isfinite(p.grad).all():
            bad.append(name)
    if missing or bad:
        raise RuntimeError(f"grad check failed. no grad: {missing[:5]} non-finite: {bad[:5]}")
    print(f"  grads OK for {sum(p.requires_grad for p in model.parameters())} trainable tensors")


def gpu_memory_report():
    if not torch.cuda.is_available():
        print("CUDA is unavailable.")
        return
    free, total = torch.cuda.mem_get_info()
    GiB = 1024 ** 3
    print(f"GPU total:         {total / GiB:.2f} GiB")
    print(f"GPU free:          {free / GiB:.2f} GiB")
    print(f"PyTorch allocated: {torch.cuda.memory_allocated() / GiB:.2f} GiB")
    print(f"PyTorch reserved:  {torch.cuda.memory_reserved() / GiB:.2f} GiB")


def clean_unused_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gpu_memory_report()
