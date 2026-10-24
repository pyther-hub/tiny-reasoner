# tiny-reasoner

Small helper library for LLM fine-tuning experiments (currently Qwen2.5-0.5B + LoRA on GSM8K).
**Experiments run in Jupyter notebooks, mostly on Kaggle GPUs.** This repo only holds clean, tested
functions that the notebooks import. Keep it simple — flat files, plain functions, no frameworks.

## Layout
```
model.py      # architecture: load_tokenizer, load_model (HF + LoRA), tiny_model (tests)
data.py       # dataset/dataloader: load_gsm8k, collate_fn, make_loader, tiny_dataset (tests)
train.py      # training loop: forward_loss, train_one_epoch, eval_val_loss, check_grads, GPU mem utils
evaluation.py # evaluation: generate_seq, extract_answer, evaluate_gsm8k, show_samples
tests/        # smoke + shape/gradient tests (CPU, tiny random model, offline)
runs/         # outputs: checkpoints, logs, predictions (gitignored)
notebooks/    # scratch analysis only — never import from here
DOCUMENTATION.md  # user-facing API reference + Kaggle usage. Update it when a public function changes.
```

## Environment
- Always use the project venv: `.venv/bin/python`, `.venv/bin/python -m pytest`. Never system python, never create another env.
- Install packages with `uv pip install --python .venv/bin/python <pkg>` and add them to `requirements.txt`.
- Local machine is a Mac (CPU). Real GPU runs happen on Kaggle — do not try to load full models or run real training locally.

## Smoke test — run after EVERY edit
```
.venv/bin/python -m pytest -q
```
Takes ~10s. Uses a tiny random Qwen2-architecture model + cached GPT-2 tokenizer, `HF_HUB_OFFLINE=1`.
Catches shape bugs, NaNs, missing grads, device errors. Do not report a change as done unless this passes.

## Debug mode
Every loop function takes `debug=False`. With `debug=True`:
- `train_one_epoch`: runs only 3 steps (`train.DEBUG_STEPS`), prints tensor shapes/dtypes/devices on the first batch, runs `check_grads`.
- `eval_val_loss`: 3 batches. `evaluate_gsm8k`: one batch, 8 new tokens.
- `forward_loss` raises `FloatingPointError` on non-finite loss (always, not only in debug).
New loops must follow the same `debug` convention.

## Rules for new code
- **Any new module / model change needs a test in `tests/`**: output-shape check, gradient check
  (all trainable params get finite grads), and an overfit-one-batch test (loss must drop sharply on a fixed batch).
- Pass `tokenizer`, `device`, `model`, `optimizer` as arguments — no globals (notebooks redefine names freely).
- Functions, not classes, unless state genuinely needs one. No config frameworks, no CLI, no YAML.
- Match the existing style: plain loops, tqdm, short docstrings only where behaviour isn't obvious.
- Module names `model`, `data` clash with common notebook variables,
  so docs and examples always use `from model import ...`, never `import model`.
- Large outputs go to `runs/<run_name>/` (local) or `/kaggle/working/runs/` (Kaggle).

## Git
- Commit small, focused changes after tests pass. Don't commit notebook outputs that contain large blobs.
