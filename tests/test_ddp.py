"""DDP on CPU (gloo, 2 processes). Workers must be module-level functions so mp.spawn can import them."""
import os

import torch
from torch.optim import AdamW

from data import tiny_dataset, make_loader
from ddp import launch, setup, cleanup, wrap_ddp, unwrap
from model import tiny_model, load_tokenizer
from train import train_one_epoch, ddp_main

TOKENIZER_NAME = "openai-community/gpt2"


def _sync_worker(rank, world_size, tmp_dir):
    device = setup(rank, world_size, timeout_s=60)
    try:
        torch.manual_seed(rank)  # different LoRA init per rank: DDP must broadcast rank 0's weights
        tokenizer = load_tokenizer(TOKENIZER_NAME)
        model = wrap_ddp(tiny_model(len(tokenizer)), device)
        loader = make_loader(tiny_dataset(8), tokenizer, batch_size=2, max_length=64, max_input_length=32,
                             distributed=True)
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
        train_one_epoch(model, loader, optimizer, device, accum_steps=2)  # 1 no_sync step + 1 synced step
        params = {n: p.detach().clone() for n, p in unwrap(model).named_parameters() if p.requires_grad}
        torch.save({"params": params, "indices": list(loader.sampler)}, os.path.join(tmp_dir, f"rank{rank}.pt"))
    finally:
        cleanup()


def test_ddp_grads_in_sync(tmp_path):
    launch(_sync_worker, str(tmp_path), nprocs=2)
    r0, r1 = (torch.load(tmp_path / f"rank{r}.pt") for r in (0, 1))

    assert not set(r0["indices"]) & set(r1["indices"]), "ranks must see different data"
    for name, p0 in r0["params"].items():
        assert torch.equal(p0, r1["params"][name]), f"{name} diverged across ranks"
    # and training actually moved the weights (lora_B starts at zero)
    assert any(p.abs().sum() > 0 for n, p in r0["params"].items() if "lora_B" in n)


def test_ddp_main_tiny(tmp_path):
    out_dir = tmp_path / "run"
    launch(ddp_main, {"tiny": True, "debug": True, "batch_size": 2, "lr": 1e-3,
                      "max_length": 64, "max_input_length": 32, "out_dir": str(out_dir)}, nprocs=2)
    assert (out_dir / "adapter_config.json").exists()
    assert (out_dir / "cfg.json").exists()
