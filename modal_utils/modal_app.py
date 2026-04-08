"""Modal app for Coconut training with async job management."""
import modal

# Modal volumes for checkpoints, data, and cache
checkpoint_volume = modal.Volume.from_name("coconut-checkpoints", create_if_missing=True)
data_volume = modal.Volume.from_name("coconut-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# Full training image with all dependencies
training_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget")  # Add wget for data preprocessing
    .pip_install(
        "torch==2.5.1",
        "numpy==2.1.3",
        "transformers==4.46.2",
        "datasets==3.1.0",
        "tqdm==4.67.0",
        "pyyaml",
        "wandb",
    )
)

# Modal app
app = modal.App("coconut-standard-training")


@app.function(
    image=training_image,
    gpu="A100-80GB:4",  # 4x A100 GPUs
    timeout=60 * 60 * 24,  # 24 hours
    secrets=[
        modal.Secret.from_name("github-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={
        "/checkpoints": checkpoint_volume,
        "/data": data_volume,
        "/root/.cache/huggingface": hf_cache,
    },
)
def train_coconut(
    config_yaml: dict,
    num_gpus: int = 4,
    repo_url: str = "https://github.com/darklord1611/repro-mech-interp.git",
    branch: str = "main",
) -> dict:
    """Train a Coconut model using the provided config.

    This function is designed to be called remotely via Modal.

    Args:
        config_yaml: Configuration dictionary (parsed YAML)
        num_gpus: Number of GPUs to use
        repo_url: GitHub repository URL
        branch: Git branch to checkout

    Returns:
        Dictionary with training results including checkpoint path
    """
    import os
    import subprocess
    import yaml
    import tempfile

    print(f"=== Coconut Training on Modal ===")
    print(f"Repository: {repo_url}")
    print(f"Branch: {branch}")
    print(f"GPUs: {num_gpus}")
    print(f"Config name: {config_yaml.get('name', 'N/A')}")

    # Get GitHub token from secret
    github_token = os.environ["GITHUB_TOKEN"]

    # Clone repository
    print("\n[1/4] Cloning repository...")
    auth_url = repo_url.replace("https://", f"https://{github_token}@")
    repo_dir = "/root/repro-mech-interp"

    try:
        subprocess.run(
            ["git", "clone", "-b", branch, "--depth", "1", auth_url, repo_dir],
            check=True,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            capture_output=True,
            text=True,
        )
        print(f"✓ Repository cloned to {repo_dir}")
    except subprocess.CalledProcessError as e:
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise

    # Change to coconut directory
    work_dir = f"{repo_dir}/coconut"
    os.chdir(work_dir)

    # Update config to use Modal volumes
    print("\n[2/4] Preparing configuration...")
    config = config_yaml.copy()

    # Update paths for Modal volumes
    config["save_path"] = "/checkpoints"

    # Write updated config
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    print(f"✓ Config updated:")
    print(f"  - save_path: /checkpoints")
    print(f"  - name: {config.get('name', 'N/A')}")
    print(f"  - epochs: {config.get('num_epochs', 'N/A')}")
    print(f"  - batch_size: {config.get('batch_size_training', 'N/A')}")
    print(f"  - c_thought: {config.get('c_thought', 'N/A')}")
    print(f"  - max_latent_stage: {config.get('max_latent_stage', 'N/A')}")

    # Run training
    print("\n[3/4] Starting training...")
    cmd = [
        "torchrun",
        "--nnodes", "1",
        "--nproc_per_node", str(num_gpus),
        "run.py",
        config_path,
    ]

    print(f"Command: {' '.join(cmd)}")

    # Run the training script (stream output in real-time)
    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    # Commit checkpoint volume
    print("\n[4/4] Committing checkpoints to volume...")
    checkpoint_volume.commit()

    # Collect results
    checkpoint_path = f"/checkpoints/{config['name']}"
    print("\n✓ Training completed successfully!")
    print(f"✓ Checkpoints saved to: {checkpoint_path}")

    return {
        "status": "success",
        "checkpoint_path": checkpoint_path,
        "config_name": config.get("name"),
        "num_epochs": config.get("num_epochs"),
    }


@app.function(
    image=training_image,
    gpu="A100-80GB:1",  # Single GPU for eval
    timeout=60 * 60 * 4,  # 4 hours
    secrets=[
        modal.Secret.from_name("github-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={
        "/checkpoints": checkpoint_volume,
        "/data": data_volume,
        "/root/.cache/huggingface": hf_cache,
    },
)
def evaluate_coconut(
    config_yaml: dict,
    checkpoint_name: str,
    repo_url: str = "https://github.com/darklord1611/repro-mech-interp.git",
    branch: str = "main",
) -> dict:
    """Evaluate a Coconut model checkpoint.

    Args:
        config_yaml: Configuration dictionary (parsed YAML)
        checkpoint_name: Name of checkpoint to evaluate (e.g., "checkpoint_10")
        repo_url: GitHub repository URL
        branch: Git branch to checkout

    Returns:
        Dictionary with evaluation results
    """
    import os
    import subprocess
    import yaml
    import tempfile

    print(f"=== Coconut Evaluation on Modal ===")
    print(f"Repository: {repo_url}")
    print(f"Branch: {branch}")
    print(f"Checkpoint: {checkpoint_name}")

    # Get GitHub token from secret
    github_token = os.environ["GITHUB_TOKEN"]

    # Clone repository
    print("\n[1/4] Cloning repository...")
    auth_url = repo_url.replace("https://", f"https://{github_token}@")
    repo_dir = "/root/repro-mech-interp"

    subprocess.run(
        ["git", "clone", "-b", branch, "--depth", "1", auth_url, repo_dir],
        check=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        capture_output=True,
    )
    print(f"✓ Repository cloned to {repo_dir}")

    # Change to coconut directory
    work_dir = f"{repo_dir}/coconut"
    os.chdir(work_dir)

    # Update config for evaluation
    print("\n[2/4] Preparing configuration...")
    config = config_yaml.copy()

    # Update paths for Modal volumes
    config["save_path"] = "/checkpoints"
    config["load_model_path"] = f"/checkpoints/{config['name']}/{checkpoint_name}"
    config["only_eval"] = True

    # Write updated config
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    print(f"✓ Config updated:")
    print(f"  - load_model_path: {config['load_model_path']}")
    print(f"  - only_eval: True")

    # Run evaluation
    print("\n[3/4] Starting evaluation...")
    cmd = [
        "torchrun",
        "--nnodes", "1",
        "--nproc_per_node", "1",
        "run.py",
        config_path,
    ]

    print(f"Command: {' '.join(cmd)}")

    # Run the evaluation script
    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed with exit code {result.returncode}")

    print("\n[4/4] Evaluation completed successfully!")

    return {
        "status": "success",
        "checkpoint": checkpoint_name,
        "config_name": config.get("name"),
    }


@app.function(
    image=training_image,
    timeout=60 * 60 * 2,  # 2 hours for preprocessing
    secrets=[modal.Secret.from_name("github-secret")],
    volumes={
        "/data": data_volume,
    },
)
def preprocess_data(
    dataset: str = "gsm8k",
    repo_url: str = "https://github.com/darklord1611/repro-mech-interp.git",
    branch: str = "main",
) -> dict:
    """Preprocess dataset by running preprocessing scripts.

    Args:
        dataset: Dataset name ("gsm8k", "prontoqa", or "prosqa")
        repo_url: GitHub repository URL
        branch: Git branch to checkout

    Returns:
        Dictionary with preprocessing results
    """
    import os
    import subprocess

    print(f"=== Data Preprocessing on Modal ===")
    print(f"Dataset: {dataset}")
    print(f"Repository: {repo_url}")
    print(f"Branch: {branch}")

    # Get GitHub token from secret
    github_token = os.environ["GITHUB_TOKEN"]

    # Clone repository
    print("\n[1/3] Cloning repository...")
    auth_url = repo_url.replace("https://", f"https://{github_token}@")
    repo_dir = "/root/repro-mech-interp"

    try:
        subprocess.run(
            ["git", "clone", "-b", branch, "--depth", "1", auth_url, repo_dir],
            check=True,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            capture_output=True,
            text=True,
        )
        print(f"✓ Repository cloned to {repo_dir}")
    except subprocess.CalledProcessError as e:
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise

    # Change to coconut directory
    work_dir = f"{repo_dir}/coconut"
    os.chdir(work_dir)

    # Run preprocessing based on dataset
    print(f"\n[2/3] Running preprocessing for {dataset}...")

    if dataset == "gsm8k":
        # Create data directory in persistent volume
        os.makedirs("/data/gsm8k", exist_ok=True)

        # Run GSM8K preprocessing
        result = subprocess.run(
            ["bash", "preprocessing/gsm_icot.bash"],
            env={**os.environ, "DATA_DIR": "/data/gsm8k"},
        )

        if result.returncode != 0:
            raise RuntimeError(f"GSM8K preprocessing failed with exit code {result.returncode}")

        # Move processed data to persistent volume
        subprocess.run(["mv", "data/gsm_train.json", "/data/gsm8k/"], check=True)
        subprocess.run(["mv", "data/gsm_valid.json", "/data/gsm8k/"], check=True)
        subprocess.run(["mv", "data/gsm_test.json", "/data/gsm8k/"], check=True)

        print("✓ GSM8K data preprocessed and saved to /data/gsm8k")
        data_path = "/data/gsm8k"

    elif dataset == "prontoqa":
        # Create data directory in persistent volume
        os.makedirs("/data/prontoqa", exist_ok=True)

        # Run ProntoQA preprocessing
        result = subprocess.run(
            ["python", "preprocessing/prontoqa.py"],
        )

        if result.returncode != 0:
            raise RuntimeError(f"ProntoQA preprocessing failed with exit code {result.returncode}")

        # Move processed data to persistent volume
        subprocess.run(["mv", "data/prontoqa_train.json", "/data/prontoqa/"], check=True)
        subprocess.run(["mv", "data/prontoqa_valid.json", "/data/prontoqa/"], check=True)
        subprocess.run(["mv", "data/prontoqa_test.json", "/data/prontoqa/"], check=True)

        print("✓ ProntoQA data preprocessed and saved to /data/prontoqa")
        data_path = "/data/prontoqa"

    elif dataset == "prosqa":
        # ProsQA data is already in the repo, just copy to persistent volume
        os.makedirs("/data/prosqa", exist_ok=True)
        subprocess.run(["cp", "data/prosqa_train.json", "/data/prosqa/"], check=True)
        subprocess.run(["cp", "data/prosqa_valid.json", "/data/prosqa/"], check=True)
        subprocess.run(["cp", "data/prosqa_test.json", "/data/prosqa/"], check=True)

        print("✓ ProsQA data copied to /data/prosqa")
        data_path = "/data/prosqa"

    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from: gsm8k, prontoqa, prosqa")

    # Commit data volume
    print("\n[3/3] Committing data to volume...")
    data_volume.commit()

    print(f"\n✓ Preprocessing completed successfully!")
    print(f"✓ Data saved to: {data_path}")

    return {
        "status": "success",
        "dataset": dataset,
        "data_path": data_path,
    }


@app.function(
    image=training_image,
    gpu="A100-80GB:4",  # 4x A100 GPUs
    timeout=60 * 60 * 24,  # 48 hours for full pipeline
    secrets=[
        modal.Secret.from_name("github-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={
        "/checkpoints": checkpoint_volume,
        "/data": data_volume,
        "/root/.cache/huggingface": hf_cache,
    },
)
def train_pipeline(
    dataset: str,
    cot_config_yaml: dict,
    coconut_config_yaml: dict,
    num_gpus: int = 4,
    skip_preprocessing: bool = False,
    skip_cot: bool = False,
    repo_url: str = "https://github.com/darklord1611/repro-mech-interp.git",
    branch: str = "main",
) -> dict:
    """Run the complete training pipeline: preprocess -> CoT -> Coconut.

    Args:
        dataset: Dataset name ("gsm8k", "prontoqa", or "prosqa")
        cot_config_yaml: Configuration dictionary for CoT training (stage 0)
        coconut_config_yaml: Configuration dictionary for Coconut training
        num_gpus: Number of GPUs to use
        skip_preprocessing: Skip preprocessing if data already exists
        skip_cot: Skip CoT training if checkpoint already exists
        repo_url: GitHub repository URL
        branch: Git branch to checkout

    Returns:
        Dictionary with pipeline results
    """
    import os
    import subprocess
    import yaml
    import tempfile
    import glob

    print(f"=== Coconut Training Pipeline on Modal ===")
    print(f"Dataset: {dataset}")
    print(f"Pipeline: preprocess -> CoT -> Coconut")
    print(f"Skip preprocessing: {skip_preprocessing}")
    print(f"Skip CoT: {skip_cot}")

    # Get GitHub token from secret
    github_token = os.environ["GITHUB_TOKEN"]

    # Clone repository
    print("\n[1/5] Cloning repository...")
    auth_url = repo_url.replace("https://", f"https://{github_token}@")
    repo_dir = "/root/repro-mech-interp"

    try:
        subprocess.run(
            ["git", "clone", "-b", branch, "--depth", "1", auth_url, repo_dir],
            check=True,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            capture_output=True,
            text=True,
        )
        print(f"✓ Repository cloned to {repo_dir}")
    except subprocess.CalledProcessError as e:
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise

    # Change to coconut directory
    work_dir = f"{repo_dir}/coconut"
    os.chdir(work_dir)

    # Step 1: Preprocess data (if needed)
    if not skip_preprocessing:
        print(f"\n[2/5] Preprocessing {dataset} data...")

        # Run preprocessing inline
        if dataset == "gsm8k":
            os.makedirs("/data/gsm8k", exist_ok=True)
            result = subprocess.run(["bash", "preprocessing/gsm_icot.bash"])
            if result.returncode != 0:
                raise RuntimeError(f"GSM8K preprocessing failed")
            subprocess.run(["mv", "data/gsm_train.json", "/data/gsm8k/"], check=True)
            subprocess.run(["mv", "data/gsm_valid.json", "/data/gsm8k/"], check=True)
            subprocess.run(["mv", "data/gsm_test.json", "/data/gsm8k/"], check=True)
        elif dataset == "prontoqa":
            os.makedirs("/data/prontoqa", exist_ok=True)
            result = subprocess.run(["python", "preprocessing/prontoqa.py"])
            if result.returncode != 0:
                raise RuntimeError(f"ProntoQA preprocessing failed")
            subprocess.run(["mv", "data/prontoqa_train.json", "/data/prontoqa/"], check=True)
            subprocess.run(["mv", "data/prontoqa_valid.json", "/data/prontoqa/"], check=True)
            subprocess.run(["mv", "data/prontoqa_test.json", "/data/prontoqa/"], check=True)
        elif dataset == "prosqa":
            os.makedirs("/data/prosqa", exist_ok=True)
            subprocess.run(["cp", "data/prosqa_train.json", "/data/prosqa/"], check=True)
            subprocess.run(["cp", "data/prosqa_valid.json", "/data/prosqa/"], check=True)
            subprocess.run(["cp", "data/prosqa_test.json", "/data/prosqa/"], check=True)

        data_volume.commit()
        print(f"✓ Preprocessing complete: /data/{dataset}")
    else:
        print("\n[2/5] Skipping preprocessing (using existing data)")

    # Step 2: Train CoT model (stage 0)
    cot_checkpoint = None
    if not skip_cot:
        print("\n[3/5] Training CoT model (stage 0)...")

        # Update CoT config paths
        cot_config = cot_config_yaml.copy()
        cot_config["save_path"] = "/checkpoints"
        cot_config["train_path"] = f"/data/{dataset}/{dataset}_train.json"
        cot_config["val_path"] = f"/data/{dataset}/{dataset}_valid.json"

        # Write CoT config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(cot_config, f)
            cot_config_path = f.name

        print(f"  CoT config: {cot_config.get('name')}")
        print(f"  Epochs: {cot_config.get('num_epochs')}")

        # Run CoT training
        cmd = [
            "torchrun",
            "--nnodes", "1",
            "--nproc_per_node", str(num_gpus),
            "run.py",
            cot_config_path,
        ]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"CoT training failed with exit code {result.returncode}")

        # Find best checkpoint
        checkpoint_dir = f"/checkpoints/{cot_config['name']}"
        checkpoints = glob.glob(f"{checkpoint_dir}/checkpoint_*")
        if not checkpoints:
            raise RuntimeError("No checkpoints found after CoT training")

        # Use the last checkpoint (assuming best is last for simplicity)
        cot_checkpoint = sorted(checkpoints)[-1]
        print(f"✓ CoT training complete: {cot_checkpoint}")
    else:
        print("\n[3/5] Skipping CoT training (using existing checkpoint)")
        # Assume checkpoint is already specified in coconut_config_yaml
        cot_checkpoint = coconut_config_yaml.get("load_model_path")

    # Commit checkpoints after CoT
    checkpoint_volume.commit()

    # Step 3: Train Coconut model
    print("\n[4/5] Training Coconut model...")

    # Update Coconut config paths
    coconut_config = coconut_config_yaml.copy()
    coconut_config["save_path"] = "/checkpoints"
    coconut_config["train_path"] = f"/data/{dataset}/{dataset}_train.json"
    coconut_config["val_path"] = f"/data/{dataset}/{dataset}_valid.json"

    # Set CoT checkpoint as initialization
    if cot_checkpoint:
        coconut_config["load_model_path"] = cot_checkpoint

    # Write Coconut config
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(coconut_config, f)
        coconut_config_path = f.name

    print(f"  Coconut config: {coconut_config.get('name')}")
    print(f"  Loading from: {coconut_config.get('load_model_path')}")
    print(f"  Epochs: {coconut_config.get('num_epochs')}")
    print(f"  Max latent stage: {coconut_config.get('max_latent_stage')}")

    # Run Coconut training
    cmd = [
        "torchrun",
        "--nnodes", "1",
        "--nproc_per_node", str(num_gpus),
        "run.py",
        coconut_config_path,
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Coconut training failed with exit code {result.returncode}")

    # Find best Coconut checkpoint
    coconut_checkpoint_dir = f"/checkpoints/{coconut_config['name']}"
    coconut_checkpoints = glob.glob(f"{coconut_checkpoint_dir}/checkpoint_*")
    if not coconut_checkpoints:
        raise RuntimeError("No checkpoints found after Coconut training")

    coconut_checkpoint = sorted(coconut_checkpoints)[-1]
    print(f"✓ Coconut training complete: {coconut_checkpoint}")

    # Commit final checkpoints
    print("\n[5/5] Committing checkpoints to volume...")
    checkpoint_volume.commit()

    print("\n✓ Training pipeline completed successfully!")
    print(f"✓ CoT checkpoint: {cot_checkpoint}")
    print(f"✓ Coconut checkpoint: {coconut_checkpoint}")

    return {
        "status": "success",
        "dataset": dataset,
        "cot_checkpoint": cot_checkpoint,
        "coconut_checkpoint": coconut_checkpoint,
        "coconut_checkpoint_dir": coconut_checkpoint_dir,
    }
