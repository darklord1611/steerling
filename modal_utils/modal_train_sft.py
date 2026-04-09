"""Modal app for Steerling SFT training."""
import modal

# Modal volumes for checkpoints and HuggingFace model cache
checkpoint_volume = modal.Volume.from_name("steerling-checkpoints", create_if_missing=True)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# Training image: Python 3.13 + PyTorch 2.8 (CUDA 12.8) + SFT deps
training_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    # PyTorch 2.8 must be fetched from the cu128 wheel index
    .run_commands(
        "pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128"
    )
    .pip_install(
        "triton>=3.0.0",
        "tiktoken~=0.8.0",
        "safetensors>=0.4.0",
        "transformers>=4.40.0,<5.0.0",
        "huggingface-hub>=0.20.0",
        "pydantic~=2.10.0",
        "numpy~=2.3.0",
        "pandas~=2.2.3",
        # SFT-specific
        "datasets>=2.0.0",
        "peft>=0.10.0",
        "tqdm",
        "wandb"
    )
)

app = modal.App("steerling-sft-training")


@app.function(
    image=training_image,
    gpu="H100:4",
    timeout=60 * 60 * 24,  # 24 hours
    secrets=[
        modal.Secret.from_name("github-secret"),     # GITHUB_TOKEN
        modal.Secret.from_name("huggingface-secret"), # HF_TOKEN
        modal.Secret.from_name("wandb-secret"),       # WANDB_API_KEY (optional)
    ],
    volumes={
        "/checkpoints": checkpoint_volume,
        "/root/.cache/huggingface": hf_cache,
    },
)
def train_sft_ddp(
    num_gpus: int = 4,
    model: str = "guidelabs/steerling-8b",
    hf_dataset_id: str = "darklord1611/tulu-3-sft-mixture-english-clean",
    max_steps: int = 44_812,
    max_seq_len: int = 2048,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    lr: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lambda_rec: float = 0.1,
    lambda_indep: float = 0.01,
    warmup_steps: int = 200,
    save_every: int = 200,
    log_every: int = 50,
    output_dir: str = "/checkpoints/sft_output",
    resume_from: str | None = None,
    wandb_project: str = "steerling-sft",
    wandb_run_name: str | None = None,
    repo_url: str = "https://github.com/darklord1611/steerling.git",
    branch: str = "main",
) -> dict:
    """Run Steerling SFT DDP training on Modal.

    Args:
        num_gpus: Number of GPUs for torchrun (must match gpu= count above).
        model: HuggingFace repo ID or local path for the base model.
        hf_dataset_id: HuggingFace dataset to train on.
        max_steps: Total gradient update steps.
        max_seq_len: Sequence length (must be divisible by 64).
        batch_size: Per-GPU batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        lr: Peak learning rate.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling.
        lora_dropout: LoRA dropout probability.
        lambda_rec: Weight for residual reconstruction loss.
        lambda_indep: Weight for independence loss.
        warmup_steps: Linear warmup steps.
        save_every: Save checkpoint every N steps.
        log_every: Log every N steps.
        output_dir: Directory to write checkpoints (inside the volume).
        resume_from: Local checkpoint dir or HF repo ID to resume from.
        repo_url: GitHub URL of the steerling repo to clone.
        branch: Git branch to checkout.

    Returns:
        Dict with status and checkpoint path.
    """
    import os
    import subprocess
    import threading
    import torch

    # Auto-detect GPU count from what Modal actually allocated
    detected_gpus = torch.cuda.device_count()
    num_gpus = detected_gpus
    effective_bs = batch_size * num_gpus * gradient_accumulation_steps

    print("=== Steerling SFT Training on Modal ===")
    print(f"Repository: {repo_url}  branch: {branch}")
    print(f"Model: {model}")
    print(f"GPUs: {num_gpus} (detected: {detected_gpus})  steps: {max_steps}  seq_len: {max_seq_len}")
    print(f"Effective batch size: {effective_bs}")

    github_token = os.environ["GITHUB_TOKEN"]

    # --- Clone repo ---
    print("\n[1/3] Cloning repository...")
    auth_url = repo_url.replace("https://", f"https://{github_token}@")
    repo_dir = "/root/steerling"

    try:
        subprocess.run(
            ["git", "clone", "-b", branch, "--depth", "1", auth_url, repo_dir],
            check=True,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            capture_output=True,
            text=True,
        )
        print(f"✓ Cloned to {repo_dir}")
    except subprocess.CalledProcessError as e:
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise

    # Install the steerling package (no-deps: image already has all deps)
    subprocess.run(
        ["pip", "install", "--no-deps", "-e", "."],
        cwd=repo_dir,
        check=True,
    )
    print("✓ steerling package installed")

    # --- Background volume committer ---
    # Commits every 5 minutes so crash recovery loses at most one save interval.
    # The training script saves every `save_every` steps; this ensures those
    # writes are persisted to the Modal volume even if the container dies.
    _stop = threading.Event()

    def _auto_commit(interval: int = 300) -> None:
        while not _stop.wait(timeout=interval):
            try:
                checkpoint_volume.commit()
                print(f"[volume] Auto-committed checkpoint volume")
            except Exception as e:
                print(f"[volume] Warning: auto-commit failed: {e}")

    _commit_thread = threading.Thread(target=_auto_commit, daemon=True)
    _commit_thread.start()

    # --- Build torchrun command ---
    print("\n[2/3] Starting DDP training...")
    cmd = [
        "torchrun",
        "--nnodes", "1",
        "--nproc_per_node", str(num_gpus),
        "scripts/sft_train_ddp.py",
        "--model", model,
        "--hf-dataset-id", hf_dataset_id,
        "--max-steps", str(max_steps),
        "--max-seq-len", str(max_seq_len),
        "--batch-size", str(batch_size),
        "--gradient-accumulation-steps", str(gradient_accumulation_steps),
        "--lr", str(lr),
        "--lora-r", str(lora_r),
        "--lora-alpha", str(lora_alpha),
        "--lora-dropout", str(lora_dropout),
        "--lambda-rec", str(lambda_rec),
        "--lambda-indep", str(lambda_indep),
        "--warmup-steps", str(warmup_steps),
        "--save-every", str(save_every),
        "--log-every", str(log_every),
        "--output-dir", output_dir,
        "--wandb-project", wandb_project,
    ]
    if resume_from:
        cmd += ["--resume-from", resume_from]
    if wandb_run_name:
        cmd += ["--wandb-run-name", wandb_run_name]

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=repo_dir)

    # --- Final commit ---
    _stop.set()
    print("\n[3/3] Committing checkpoints...")
    checkpoint_volume.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    final_ckpt = f"{output_dir}/final"
    print(f"✓ Training complete. Checkpoint: {final_ckpt}")
    return {"status": "success", "checkpoint_path": final_ckpt}


@app.function(
    image=training_image,
    gpu="H100:1",
    timeout=60 * 60 * 12,  # 12 hours
    secrets=[
        modal.Secret.from_name("github-secret"),
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={
        "/checkpoints": checkpoint_volume,
        "/root/.cache/huggingface": hf_cache,
    },
)
def train_sft_single(
    model: str = "guidelabs/steerling-8b",
    hf_dataset_id: str = "darklord1611/tulu-3-sft-mixture-english-clean",
    max_steps: int = 716_987,
    max_seq_len: int = 2048,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    lr: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lambda_rec: float = 0.1,
    lambda_indep: float = 0.01,
    warmup_steps: int = 100,
    save_every: int = 500,
    log_every: int = 10,
    output_dir: str = "/checkpoints/sft_output_single",
    resume_from: str | None = None,
    wandb_project: str = "steerling-sft",
    wandb_run_name: str | None = None,
    repo_url: str = "https://github.com/darklord1611/steerling.git",
    branch: str = "main",
) -> dict:
    """Run Steerling SFT training on a single GPU (for testing / small runs).

    Args:
        model: HuggingFace repo ID or local path for the base model.
        hf_dataset_id: HuggingFace dataset to train on.
        max_steps: Total gradient update steps.
        max_seq_len: Sequence length (must be divisible by 64).
        batch_size: Batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        lr: Peak learning rate.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling.
        lora_dropout: LoRA dropout probability.
        lambda_rec: Weight for residual reconstruction loss.
        lambda_indep: Weight for independence loss.
        warmup_steps: Linear warmup steps.
        save_every: Save checkpoint every N steps.
        log_every: Log every N steps.
        output_dir: Directory to write checkpoints (inside the volume).
        resume_from: Local checkpoint dir or HF repo ID to resume from.
        repo_url: GitHub URL of the steerling repo to clone.
        branch: Git branch to checkout.

    Returns:
        Dict with status and checkpoint path.
    """
    import os
    import subprocess
    import threading

    print("=== Steerling SFT Training on Modal (single GPU) ===")
    print(f"Repository: {repo_url}  branch: {branch}")
    print(f"Model: {model}  steps: {max_steps}")

    github_token = os.environ["GITHUB_TOKEN"]

    # --- Clone repo ---
    print("\n[1/3] Cloning repository...")
    auth_url = repo_url.replace("https://", f"https://{github_token}@")
    repo_dir = "/root/steerling"

    try:
        subprocess.run(
            ["git", "clone", "-b", branch, "--depth", "1", auth_url, repo_dir],
            check=True,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            capture_output=True,
            text=True,
        )
        print(f"✓ Cloned to {repo_dir}")
    except subprocess.CalledProcessError as e:
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise

    subprocess.run(
        ["pip", "install", "--no-deps", "-e", "."],
        cwd=repo_dir,
        check=True,
    )
    print("✓ steerling package installed")

    # --- Build command ---
    print("\n[2/3] Starting training...")
    cmd = [
        "python", "scripts/sft_train.py",
        "--model", model,
        "--hf-dataset-id", hf_dataset_id,
        "--max-steps", str(max_steps),
        "--max-seq-len", str(max_seq_len),
        "--batch-size", str(batch_size),
        "--gradient-accumulation-steps", str(gradient_accumulation_steps),
        "--lr", str(lr),
        "--lora-r", str(lora_r),
        "--lora-alpha", str(lora_alpha),
        "--lora-dropout", str(lora_dropout),
        "--lambda-rec", str(lambda_rec),
        "--lambda-indep", str(lambda_indep),
        "--warmup-steps", str(warmup_steps),
        "--save-every", str(save_every),
        "--log-every", str(log_every),
        "--output-dir", output_dir,
        "--wandb-project", wandb_project,
        "--wandb",
    ]
    if resume_from:
        cmd += ["--resume-from", resume_from]
    if wandb_run_name:
        cmd += ["--wandb-run-name", wandb_run_name]

    # --- Background volume committer ---
    _stop = threading.Event()

    def _auto_commit(interval: int = 300) -> None:
        while not _stop.wait(timeout=interval):
            try:
                checkpoint_volume.commit()
                print("[volume] Auto-committed checkpoint volume")
            except Exception as e:
                print(f"[volume] Warning: auto-commit failed: {e}")

    threading.Thread(target=_auto_commit, daemon=True).start()

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=repo_dir)

    # --- Final commit ---
    _stop.set()
    print("\n[3/3] Committing checkpoints...")
    checkpoint_volume.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    final_ckpt = f"{output_dir}/final"
    print(f"✓ Training complete. Checkpoint: {final_ckpt}")
    return {"status": "success", "checkpoint_path": final_ckpt}
