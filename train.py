import gc
import json
import os
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from tqdm.auto import tqdm

from data import load_gsm8k, make_loader, tiny_dataset
from ddp import setup, cleanup, barrier, is_main, reduce_mean, wrap_ddp, unwrap
from model import load_tokenizer, load_model, tiny_model

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
    """debug=True: run DEBUG_STEPS batches, print tensor shapes on the first, check grads.
    Works with a DDP-wrapped model: grads are only all-reduced on update steps. Returns this rank's losses."""
    model.train()
    optimizer.zero_grad(set_to_none=True) #set gradients to None instead of zeroing them out, which can be more efficient
    num_batches = min(len(train_loader), DEBUG_STEPS) if debug else len(train_loader)
    if debug:
        accum_steps = 1
    losses = []

    with tqdm(enumerate(train_loader), total=num_batches, disable=debug or not is_main()) as pbar:
        for batch_idx, batch in pbar:
            if batch_idx >= num_batches:
                break
            first = debug and batch_idx == 0
            if first and is_main():
                print("[debug] first batch:")

            is_update_step = (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == num_batches
            sync = nullcontext() if is_update_step or not isinstance(model, DDP) else model.no_sync()
            with sync:
                loss, _ = forward_loss(model, batch, device, debug=first and is_main())
                (loss / accum_steps).backward()
            if first:
                check_grads(model)

            if is_update_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item())
            if debug and is_main():
                print(f"[debug] step {batch_idx + 1}/{num_batches} loss={loss.item():.4f}")
            pbar.set_postfix({"train_loss": loss.item()})
    return losses


@torch.no_grad()
def eval_val_loss(model, val_loader, device, debug=False):
    """Mean val loss, averaged across ranks under DDP."""
    model.eval()
    num_batches = min(len(val_loader), DEBUG_STEPS) if debug else len(val_loader)
    total_loss = 0.0
    with tqdm(enumerate(val_loader), total=num_batches, disable=debug or not is_main()) as pbar:
        for batch_idx, batch in pbar:
            if batch_idx >= num_batches:
                break
            loss, _ = forward_loss(model, batch, device)
            total_loss += loss.item()
            pbar.set_postfix({"val_loss": total_loss / (batch_idx + 1)})
    return reduce_mean(total_loss / num_batches, device)


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
    if is_main():
        print(f"  grads OK for {sum(p.requires_grad for p in model.parameters())} trainable tensors")


DDP_DEFAULTS = dict(
    model_name="Qwen/Qwen2.5-0.5B",
    out_dir="runs/ddp",
    batch_size=6,            # per GPU: effective batch = batch_size * world_size * accum_steps
    accum_steps=1,
    lr=3e-5,
    epochs=1,
    max_length=384,
    max_input_length=128,
    val_size=256,
    grad_ckpt=True,
    lora_kwargs={},
    seed=42,
    debug=False,
    tiny=False,              # tiny random model + fake data (tests / CPU)
)


def ddp_main(rank, world_size, cfg):
    """One DDP worker. From a notebook: ddp.launch(ddp_main, cfg).
    Processes share nothing with the notebook, so this builds everything itself
    and rank 0 saves the adapter + cfg.json to cfg["out_dir"]; reload it with model.load_adapter."""
    cfg = {**DDP_DEFAULTS, **cfg}
    debug = cfg["debug"]
    device = setup(rank, world_size, timeout_s=120 if debug else 1800)
    try:
        torch.manual_seed(cfg["seed"])
        # rank 0 downloads model + data first; the others then read from the HF cache
        if not is_main():
            barrier()
        if cfg["tiny"]:
            tokenizer = load_tokenizer("openai-community/gpt2")
            model = tiny_model(len(tokenizer)).to(device)
            train_ds = val_ds = tiny_dataset(16)
        else:
            tokenizer = load_tokenizer(cfg["model_name"])
            model = load_model(cfg["model_name"], device, grad_ckpt=cfg["grad_ckpt"], **cfg["lora_kwargs"])
            train_ds, val_ds, _ = load_gsm8k(val_size=cfg["val_size"], seed=cfg["seed"])
        if is_main():
            barrier()

        model = wrap_ddp(model, device)
        loader_kwargs = dict(max_length=cfg["max_length"], max_input_length=cfg["max_input_length"], distributed=True)
        train_loader = make_loader(train_ds, tokenizer, cfg["batch_size"], shuffle=True, **loader_kwargs)
        val_loader = make_loader(val_ds, tokenizer, cfg["batch_size"], shuffle=False, **loader_kwargs)
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])
        if is_main():
            print(f"[ddp] world_size={world_size} device={device} "
                  f"effective batch={cfg['batch_size'] * world_size * cfg['accum_steps']}")

        for epoch in range(cfg["epochs"]):
            train_loader.sampler.set_epoch(epoch)
            losses = train_one_epoch(model, train_loader, optimizer, device, cfg["accum_steps"], debug=debug)
            train_loss = reduce_mean(sum(losses) / len(losses), device)
            val_loss = eval_val_loss(model, val_loader, device, debug=debug)
            if is_main():
                print(f"[ddp] epoch {epoch + 1}/{cfg['epochs']} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if is_main():
            os.makedirs(cfg["out_dir"], exist_ok=True)
            unwrap(model).save_pretrained(cfg["out_dir"])
            with open(os.path.join(cfg["out_dir"], "cfg.json"), "w") as f:
                json.dump(cfg, f, indent=2)
            print(f"[ddp] saved to {cfg['out_dir']}")
        barrier()
    finally:
        cleanup()


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
