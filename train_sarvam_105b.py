import modal

app = modal.App("sarvam-105b-uncensored")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "peft",
        "trl",
        "datasets",
        "huggingface_hub",
        "safetensors",
        "bitsandbytes>=0.43.0",
        "accelerate",
        "transformers==4.51.3",  # pin here
    ], extra_options="--no-build-isolation")
)

@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 60 * 24,
    memory=131072,
)
def train():
    import json, os, glob, zipfile, urllib.request
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import Dataset
    from huggingface_hub import login, create_repo

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    MODEL_PATH   = "sarvamai/sarvam-105b"
    OUTPUT_PATH  = "/root/sarvam-105b-uncensored"
    DATA_ZIP_URL = "https://github.com/aloshdenny/sarvam/raw/main/reddit_dark_posts.zip"
    DATA_DIR     = "/root/reddit_dark_posts"
    MAX_SEQ_LEN  = 1024
    HF_TOKEN     = ""
    REPO_ID      = "aoxo/sarvam-105b-uncensored"

    # ── 1. Download & unzip dataset ───────────────────────────────────────────
    print("Downloading dataset...")
    zip_path = "/root/reddit_dark_posts.zip"
    urllib.request.urlretrieve(DATA_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("/root")
    print(f"Unzipped to {DATA_DIR}")

    # ── 2. Load dataset ───────────────────────────────────────────────────────
    raw = []
    for fpath in glob.glob(os.path.join(DATA_DIR, "*.json")):
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read().strip()
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                raw.append(data)
            elif isinstance(data, list):
                raw.extend(data)
        except json.JSONDecodeError:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw.append(json.loads(line))
                    except:
                        pass

    print(f"Loaded {len(raw)} raw items")

    examples = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if "prompt" not in item or "response" not in item:
            continue
        prompt = str(item["prompt"]).strip()
        responses = item["response"]
        if not isinstance(responses, list):
            responses = [responses]
        for resp in responses:
            resp = str(resp).strip()
            if prompt and resp:
                examples.append({"prompt": prompt, "response": resp})

    print(f"Expanded to {len(examples)} training examples")

    # ── 3. Tokenizer + formatting ─────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def format_example(ex):
        return {
            "text": tokenizer.apply_chat_template(
                [
                    {"role": "user",      "content": ex["prompt"]},
                    {"role": "assistant", "content": ex["response"]},
                ],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        }

    formatted = [format_example(e) for e in examples]

    def token_len(text):
        return len(tokenizer(text, truncation=False)["input_ids"])

    filtered = [f for f in formatted if token_len(f["text"]) <= MAX_SEQ_LEN]
    print(f"After length filter: {len(filtered)} examples")

    dataset = Dataset.from_list(filtered).train_test_split(test_size=0.1, seed=42)
    print(f"Train: {len(dataset['train'])}  Val: {len(dataset['test'])}")

    # ── 4. QLoRA: 4-bit NF4 quantization config ───────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # NormalFloat4 — best for LLM weights
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,       # nested quantization saves ~0.4 bits/param extra
    )

    # ── 5. Load model in 4-bit ────────────────────────────────────────────────
    print("Loading model in 4-bit NF4...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",                   # single GPU — maps everything to cuda:0
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    # Required for QLoRA: casts layernorms to fp32, enables grad checkpointing safely
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # ── 6. LoRA config (lower r than full LoRA — 4-bit already compresses) ───
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,                  # reduced from 64; adapters are fp32 anyway
        lora_alpha=128,         # keep alpha = 2×r
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # attention
            "gate_proj", "up_proj", "down_proj",       # MLP
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Sarvam-105B gate asserts `not self.training` — keep all gate modules frozen in eval mode
    for name, module in model.named_modules():
        if "gate" in name.lower():
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    # Hook to re-enforce eval on gate modules every forward pass
    # (gradient checkpointing can flip modules back to train mode)
    def _keep_gates_eval(module, input):
        pass

    gate_modules = [m for n, m in model.named_modules() if "gate" in n.lower()]

    def make_gate_eval_hook(gates):
        def hook(module, input, output):
            for g in gates:
                g.eval()
        return hook

    # Attach to the top-level model so it fires before every forward
    for name, module in model.named_modules():
        if "gate" in name.lower():
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
            # Permanently disable train mode on gate modules
            module.train = lambda self=module, mode=True: self  # no-op

    # ── 7. Training args ──────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir="/root/tmp_checkpoints",
        num_train_epochs=1,
        per_device_train_batch_size=4,       # lower than 30B run — 105B even quantized is large
        gradient_accumulation_steps=8,       # effective batch = 32, same as before
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        optim="paged_adamw_8bit",            # QLoRA-native optimizer — critical for memory
        eval_strategy="no",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=8,            # reduced — memory pressure
        dataloader_pin_memory=True,
        max_grad_norm=0.3,                   # QLoRA paper recommendation
        group_by_length=True,                # packs similar-length sequences → less padding waste
    )

    # ── 8. Train ──────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
    )

    print("Starting QLoRA training...")
    trainer.train()
    print("Training complete.")

    # ── 9. Merge + unload (on-the-fly, like 30B) ─────────────────────────────
    print("Merging LoRA weights...")
    merged_model = trainer.model.merge_and_unload()
    print("Merge complete.")

    # ── 10. Save merged model (manual sharding) ───────────────────────────────
    print(f"Saving merged model to {OUTPUT_PATH}...")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    merged_model.config.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)

    from safetensors.torch import save_file

    state_dict = merged_model.state_dict()
    shard_size = 9 * 1024**3
    current_shard, current_size, shard_idx, index = {}, 0, 1, {}

    for key, tensor in state_dict.items():
        tensor_size = tensor.nbytes
        if current_size + tensor_size > shard_size and current_shard:
            shard_name = f"model-{shard_idx:05d}.safetensors"
            save_file(current_shard, os.path.join(OUTPUT_PATH, shard_name))
            for k in current_shard:
                index[k] = shard_name
            current_shard, current_size = {}, 0
            shard_idx += 1
        current_shard[key] = tensor.contiguous()
        current_size += tensor_size

    if current_shard:
        shard_name = f"model-{shard_idx:05d}.safetensors"
        save_file(current_shard, os.path.join(OUTPUT_PATH, shard_name))
        for k in current_shard:
            index[k] = shard_name

    with open(os.path.join(OUTPUT_PATH, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": index}, f, indent=2)

    print("✓ Merged model saved.")

    # ── 11. Push to HuggingFace ───────────────────────────────────────────────
    print(f"\nPushing to HuggingFace: {REPO_ID}")
    login(token=HF_TOKEN)
    create_repo(REPO_ID, exist_ok=True)

    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_folder(folder_path=OUTPUT_PATH, repo_id=REPO_ID, repo_type="model")
    tokenizer.push_to_hub(REPO_ID)
    print(f"✓ Uploaded to https://huggingface.co/{REPO_ID}")


@app.local_entrypoint()
def main():
    train.remote()