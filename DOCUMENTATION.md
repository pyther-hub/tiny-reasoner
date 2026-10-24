# once-llm — Documentation

Helper code for LLM fine-tuning experiments. You run experiments in notebooks (Kaggle GPU or local);
the notebook imports functions from this repo.

---

## 1. Using it on Kaggle

Notebook settings: **Internet ON**, accelerator GPU.

```python
# cell 1 — setup (re-run to pull latest code)
import os, sys
REPO = "/kaggle/working/once-llm"
if not os.path.exists(REPO):
    !git clone https://github.com/<you>/once-llm.git {REPO}
else:
    !git -C {REPO} pull
sys.path.insert(0, REPO)
!pip install -q peft

%load_ext autoreload
%autoreload 2
```

**Private repo:** add a Kaggle Secret named `GH_TOKEN` (Add-ons → Secrets), then:
```python
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GH_TOKEN")
!git clone https://{token}@github.com/<you>/once-llm.git {REPO}
```

Save outputs to `/kaggle/working/runs/<run_name>/` — that directory is kept in the notebook's output.

## 2. Using it locally

```bash
source .venv/bin/activate          # always this env
python -m pytest -q                # smoke tests, ~10s, run after every edit
```
In a local notebook, put `sys.path.insert(0, "..")` in the setup cell if the notebook is in `notebooks/`.

---

## 3. Typical notebook

```python
import torch
from torch.optim import AdamW
from model import get_device, load_tokenizer, load_model
from data import load_gsm8k, make_loader
from train import train_one_epoch, eval_val_loss, clean_unused_gpu_memory
from eval import evaluate_gsm8k, show_samples

DEBUG = True          # True: 3 steps, shape printing, grad check. Flip to False for the real run.
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
device = get_device()

tokenizer = load_tokenizer(MODEL_NAME)
llm = load_model(MODEL_NAME, device, lora=True, grad_ckpt=True, r=8, lora_alpha=16)

train_ds, val_ds, test_ds = load_gsm8k(val_size=256)
train_loader = make_loader(train_ds, tokenizer, batch_size=6)
val_loader = make_loader(val_ds, tokenizer, batch_size=6, shuffle=False)

optimizer = AdamW([p for p in llm.parameters() if p.requires_grad], lr=3e-5)

for epoch in range(1 if DEBUG else 5):
    train_one_epoch(llm, train_loader, optimizer, device, accum_steps=4, debug=DEBUG)
    print("val loss:", eval_val_loss(llm, val_loader, device, debug=DEBUG))
    show_samples(llm, tokenizer, test_ds, device, n=3)
    clean_unused_gpu_memory()

acc, preds = evaluate_gsm8k(llm, tokenizer, test_ds, device, debug=DEBUG)
```

> Always do a `DEBUG = True` pass first. It takes seconds and catches shape bugs, NaNs and device errors
> before you burn GPU quota.

> Name the model variable something other than `model` (e.g. `llm`) to avoid confusion with the `model` module.
> Never `import eval` — use `from eval import ...` (it would shadow Python's `eval`).

---

## 4. API reference

### `model.py`
| Function | What it does |
|---|---|
| `get_device()` | `cuda:0` if available, else `cpu`. |
| `load_tokenizer(model_name)` | HF tokenizer; sets `pad_token = eos_token` if missing. |
| `load_model(model_name, device, dtype=float32, lora=True, grad_ckpt=False, **lora_kwargs)` | Loads `AutoModelForCausalLM` (sdpa attention), optionally wraps with LoRA and enables gradient checkpointing, moves to device, prints trainable params. |
| `add_lora(model, r=8, lora_alpha=16, lora_dropout=0.05, target_modules=("q_proj","v_proj"))` | Wraps a model with a peft LoRA adapter. |
| `tiny_model(vocab_size, lora=True)` | 2-layer, hidden-64 random Qwen2 model for tests. No download. |
| `print_trainable_parameters(model)` | Trainable / total param counts. |

### `data.py`
| Function | What it does |
|---|---|
| `load_gsm8k(val_size=256, seed=42)` | Returns `(train, val, test)` HF datasets; val is split from train. |
| `format_prompt(question)` | `"Question: {q}\nAnswer:"` — used by both training and generation. |
| `collate_fn(batch, tokenizer, max_length, max_input_length)` | Tokenizes prompt + answer + EOS, right-pads. Prompt and padding labels are `-100`. Returns `input_ids`, `attention_mask`, `labels` (all `[B, T]` long). |
| `make_loader(dataset, tokenizer, batch_size, max_length=384, max_input_length=128, shuffle=True)` | DataLoader using `collate_fn`. |
| `tiny_dataset(n=8)` | Fake GSM8K-format rows for tests. |

### `train.py`
| Function | What it does |
|---|---|
| `forward_loss(model, batch, device, debug=False)` | Forward pass + shifted next-token cross-entropy. Returns `(loss, logits)`. Raises on NaN/inf loss. `debug` prints all tensor shapes. |
| `train_one_epoch(model, loader, optimizer, device, accum_steps=4, debug=False)` | One epoch with gradient accumulation. Returns list of per-batch losses. `debug`: 3 steps, no accumulation, shape print + grad check. |
| `eval_val_loss(model, loader, device, debug=False)` | Mean val loss (no grad). |
| `check_grads(model)` | Raises if any trainable param has no grad or a non-finite grad. |
| `calc_loss(logits, labels)` | Flat cross-entropy, `ignore_index=-100`. |
| `gpu_memory_report()` / `clean_unused_gpu_memory()` | GPU memory stats / gc + empty cache. |

### `eval.py`
| Function | What it does |
|---|---|
| `generate_seq(model, tokenizer, questions, device, max_seq_len=512)` | Greedy batched decoding (left-padded, no KV cache). Returns generated text only. |
| `extract_answer(text)` | Last `#### <number>` in the text, commas stripped, or `None`. |
| `evaluate_gsm8k(model, tokenizer, dataset, device, batch_size=8, max_new_tokens=256, debug=False)` | Exact-match accuracy. Returns `(accuracy, predictions)`. |
| `show_samples(model, tokenizer, dataset, device, n=5, max_new_tokens=64)` | Prints question / generated / expected. |

---

## 5. Tests (`tests/test_smoke.py`)

| Test | Catches |
|---|---|
| `test_collate_shapes` | Wrong shapes, padding not masked, rows with no answer tokens. |
| `test_forward_shapes` | Logits shape `[B, T, V]`, finite scalar loss. |
| `test_lora_grads` | Only LoRA params trainable, all of them receive finite grads. |
| `test_overfit_one_batch` | Model + loss can actually learn (loss drops >70% on one batch). |
| `test_debug_pipeline` | Full train → val → eval path runs end-to-end in debug mode. |
| `test_extract_answer` | Answer parsing edge cases. |

Adding a new module? Add a shape test, a gradient test and an overfit-one-batch test for it.

## 6. Workflow
1. Edit code locally (or have Claude Code edit it) → `python -m pytest -q` → commit → `git push`.
2. On Kaggle: re-run the setup cell (`git pull`), run with `DEBUG = True`, then `DEBUG = False`.
3. Prototype new ideas inside the notebook freely; once a function is stable, move it into the repo with a test.
