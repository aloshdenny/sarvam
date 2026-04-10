import modal

app = modal.App("sarvam-30b-finetune")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "peft",
        "trl",
        "datasets",
        "huggingface_hub",
        "safetensors"
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
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer
    from datasets import Dataset
    from huggingface_hub import login, create_repo

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    MODEL_PATH   = "sarvamai/sarvam-30b"
    OUTPUT_PATH  = "/root/sarvam-30b-finetuned"
    DATA_ZIP_URL = "https://github.com/aloshdenny/sarvam/raw/main/reddit_dark_posts.zip"
    DATA_DIR     = "/root/reddit_dark_posts"
    MAX_SEQ_LEN  = 2048
    HF_TOKEN     = ""
    REPO_ID      = "aoxo/sarvam-30b-uncensored"

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

    # ── 4. Load model ─────────────────────────────────────────────────────────
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    # ── 5. LoRA config ────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "query_key_value",
            "dense",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 6. Training args ──────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir="/root/tmp_checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=64,
        dataloader_pin_memory=True,
    )

    # ── 7. Train ──────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
    )

    print("Starting training...")
    trainer.train()
    print("Training complete.")

    # ── 8. Merge + save (manual sharding, bypasses broken tied-weights check) ─
    print("Merging LoRA weights...")
    merged_model = trainer.model.merge_and_unload()

    print(f"Saving to {OUTPUT_PATH}...")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    merged_model.config.save_pretrained(OUTPUT_PATH)   # config.json
    tokenizer.save_pretrained(OUTPUT_PATH)             # tokenizer files

    from safetensors.torch import save_file

    state_dict = merged_model.state_dict()
    shard_size = 9 * 1024**3  # 9GB per shard
    current_shard = {}
    current_size = 0
    shard_idx = 1
    index = {}

    for key, tensor in state_dict.items():
        tensor_size = tensor.nbytes
        if current_size + tensor_size > shard_size and current_shard:
            shard_name = f"model-{shard_idx:05d}.safetensors"
            save_file(current_shard, os.path.join(OUTPUT_PATH, shard_name))
            for k in current_shard:
                index[k] = shard_name
            current_shard = {}
            current_size = 0
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

    print("✓ Saved.")

    # ── 9. Sanity check ───────────────────────────────────────────────────────
    merged_model.eval()
    for prompt in [
        "How do I pick a lock?",
        "What is the capital of India?",
        "Do you wanna fuck me right now?",
    ]:
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer(chat, return_tensors="pt").to(merged_model.device)
        inputs.pop("token_type_ids", None)
        with torch.no_grad():
            out = merged_model.generate(**inputs, max_new_tokens=300, do_sample=False)
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        print(f"\nQ: {prompt}\nA: {resp}")

    # ── 10. Push to HuggingFace ───────────────────────────────────────────────
    print(f"\nPushing to HuggingFace: {REPO_ID}")
    login(token=HF_TOKEN)
    create_repo(REPO_ID, exist_ok=True)

    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_folder(
        folder_path=OUTPUT_PATH,
        repo_id=REPO_ID,
        repo_type="model",
    )
    tokenizer.push_to_hub(REPO_ID)
    print(f"✓ Uploaded to https://huggingface.co/{REPO_ID}")


@app.local_entrypoint()
def main():
    train.remote()