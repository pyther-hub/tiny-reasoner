import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2Config
from peft import LoraConfig, TaskType, get_peft_model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def add_lora(model, r=8, lora_alpha=16, lora_dropout=0.05, target_modules=("q_proj", "v_proj")):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(target_modules),
        bias="none",
    )
    return get_peft_model(model, lora_config)


def load_model(model_name, device, dtype=torch.float32, lora=True, grad_ckpt=False, **lora_kwargs):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    if lora:
        model = add_lora(model, **lora_kwargs)
    if grad_ckpt:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    model.to(device)
    print_trainable_parameters(model)
    return model


def tiny_model(vocab_size, lora=True):
    """Randomly initialised Qwen2-architecture model (~1M params). For smoke tests only, no download."""
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    model = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa")
    if lora:
        model = add_lora(model)
    return model


def print_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")
