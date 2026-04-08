"""Services for managing multiple Modal Coconut training jobs."""
import json
import yaml
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

import modal

# Directory for storing job status
JOBS_DIR = Path(__file__).parent / "modal_jobs"
JOBS_DIR.mkdir(exist_ok=True, parents=True)

# Modal app name (must match app name in modal_app.py)
APP_NAME = "coconut-standard-training"


def compute_config_hash(config: Dict[str, Any], num_gpus: int) -> str:
    """Compute a unique hash for a training configuration.

    Args:
        config: Training configuration dictionary
        num_gpus: Number of GPUs

    Returns:
        8-character hash string
    """
    # Select important config keys for hashing
    important_keys = [
        "name", "num_epochs", "batch_size_training", "learning_rate",
        "max_latent_stage", "epochs_per_stage", "c_thought",
        "coconut", "cot", "no_thoughts", "no_cot",
        "reset_optimizer", "pad_latent_to_max"
    ]

    # Build a canonical representation
    config_subset = {k: config.get(k) for k in important_keys if k in config}
    config_subset["num_gpus"] = num_gpus

    # Create hash
    config_str = json.dumps(config_subset, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:8]


@dataclass
class JobStatus:
    """Status of a training job."""
    job_id: str
    config_hash: str
    config_name: str
    config_yaml_path: str
    num_gpus: int
    status: str  # "pending", "running", "completed", "failed"
    function_call_id: Optional[str] = None
    checkpoint_path: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "job_id": self.job_id,
            "config_hash": self.config_hash,
            "config_name": self.config_name,
            "config_yaml_path": self.config_yaml_path,
            "num_gpus": self.num_gpus,
            "status": self.status,
            "function_call_id": self.function_call_id,
            "checkpoint_path": self.checkpoint_path,
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }

    @staticmethod
    def from_dict(data: dict) -> "JobStatus":
        """Create from dictionary."""
        return JobStatus(
            job_id=data["job_id"],
            config_hash=data.get("config_hash", "unknown"),
            config_name=data["config_name"],
            config_yaml_path=data["config_yaml_path"],
            num_gpus=data["num_gpus"],
            status=data["status"],
            function_call_id=data.get("function_call_id"),
            checkpoint_path=data.get("checkpoint_path"),
            error=data.get("error"),
            created_at=data.get("created_at"),
            completed_at=data.get("completed_at"),
        )


def ensure_app_deployed() -> None:
    """Ensure the Modal app is deployed (idempotent)."""
    try:
        modal.App.lookup(APP_NAME)
        print(f"✓ App '{APP_NAME}' already deployed")
    except Exception:
        print(f"Deploying app '{APP_NAME}'...")
        from modal_utils.modal_app import app
        with modal.enable_output():
            app.deploy()
        print(f"✓ App '{APP_NAME}' deployed successfully")


def _get_job_file(job_id: str) -> Path:
    """Get the status file path for a job."""
    return JOBS_DIR / f"{job_id}.json"


def _save_job_status(status: JobStatus):
    """Save job status to file."""
    job_file = _get_job_file(status.job_id)
    with open(job_file, 'w') as f:
        json.dump(status.to_dict(), f, indent=2)


def _load_job_status(job_id: str) -> Optional[JobStatus]:
    """Load job status from file."""
    job_file = _get_job_file(job_id)
    if not job_file.exists():
        return None
    with open(job_file, 'r') as f:
        data = json.load(f)
    return JobStatus.from_dict(data)


def list_all_jobs() -> List[JobStatus]:
    """List all tracked jobs.

    Returns:
        List of job statuses, sorted by creation time (newest first)
    """
    jobs = []
    for job_file in JOBS_DIR.glob("*.json"):
        try:
            with open(job_file, 'r') as f:
                data = json.load(f)
            jobs.append(JobStatus.from_dict(data))
        except Exception as e:
            print(f"Warning: Could not load {job_file}: {e}")

    # Sort by creation time (newest first)
    jobs.sort(key=lambda j: j.created_at or "", reverse=True)
    return jobs


def get_jobs_by_config(config_hash: str) -> List[JobStatus]:
    """Get all jobs matching a config hash.

    Args:
        config_hash: Configuration hash

    Returns:
        List of matching jobs, sorted by creation time (newest first)
    """
    all_jobs = list_all_jobs()
    matching = [j for j in all_jobs if j.config_hash == config_hash]
    return matching


async def submit_job(
    config_yaml_path: str,
    num_gpus: int = 4,
    repo_url: str = "https://github.com/darklord1611/repro-mech-interp.git",
    branch: str = "main",
) -> JobStatus:
    """Submit a training job asynchronously.

    Args:
        config_yaml_path: Path to config YAML file
        num_gpus: Number of GPUs to use
        repo_url: GitHub repository URL
        branch: Git branch to checkout

    Returns:
        Job status with function_call_id for tracking
    """
    # Load config to get name
    with open(config_yaml_path) as f:
        config_yaml = yaml.safe_load(f)
    config_name = config_yaml.get("name", "coconut")

    # Compute config hash
    config_hash = compute_config_hash(config_yaml, num_gpus)

    # Check for existing jobs with same config
    existing_jobs = get_jobs_by_config(config_hash)
    active_jobs = [j for j in existing_jobs if j.status in ["pending", "running"]]

    if active_jobs:
        print(f"\n⚠ Warning: Found {len(active_jobs)} active job(s) with same config:")
        for job in active_jobs:
            print(f"  - {job.job_id} ({job.status})")
        print()

    # Generate job ID with config hash
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = f"{config_name}_{config_hash}_{timestamp}"

    print(f"Submitting job: {job_id}")
    print(f"  Config: {config_yaml_path}")
    print(f"  Config Hash: {config_hash}")
    print(f"  GPUs: {num_gpus}")

    # Ensure app is deployed
    ensure_app_deployed()

    # Get function reference
    train_func = modal.Function.from_name(APP_NAME, "train_coconut")

    # Spawn async job
    print("Spawning training job...")
    with modal.enable_output():
        function_call = train_func.spawn(
            config_yaml=config_yaml,
            num_gpus=num_gpus,
            repo_url=repo_url,
            branch=branch,
        )

    # Create and save status
    status = JobStatus(
        job_id=job_id,
        config_hash=config_hash,
        config_name=config_name,
        config_yaml_path=config_yaml_path,
        num_gpus=num_gpus,
        status="pending",
        function_call_id=function_call.object_id,
        created_at=datetime.now().isoformat(),
    )
    _save_job_status(status)

    print(f"✓ Job submitted: {job_id}")
    print(f"  Function Call ID: {function_call.object_id}")

    return status


async def get_job_status(job_id: Optional[str] = None) -> Optional[JobStatus]:
    """Check status of a specific job or list all jobs.

    Args:
        job_id: Specific job ID to check. If None, returns the most recent job.

    Returns:
        Job status, or None if no job exists
    """
    # If no job_id specified, get most recent job
    if job_id is None:
        all_jobs = list_all_jobs()
        if not all_jobs:
            return None
        status = all_jobs[0]  # Most recent
        job_id = status.job_id
        print(f"Checking most recent job: {job_id}")
    else:
        status = _load_job_status(job_id)
        if status is None:
            return None

    # If already completed or failed, return cached status
    if status.status in ["completed", "failed"]:
        return status

    # Poll Modal for current status
    try:
        print(f"Checking job {status.job_id}...")

        # Get FunctionCall object
        function_call = modal.FunctionCall.from_id(status.function_call_id)

        # Check if completed (non-blocking)
        try:
            result = function_call.get(timeout=0)

            # Job completed
            status.status = "completed"
            status.checkpoint_path = result.get("checkpoint_path")
            status.completed_at = datetime.now().isoformat()
            _save_job_status(status)

            print(f"✓ Job completed: {status.checkpoint_path}")

        except TimeoutError:
            # Still running
            if status.status == "pending":
                status.status = "running"
                _save_job_status(status)
            print(f"Job is {status.status}")

    except Exception as e:
        # Job failed
        print(f"✗ Job failed: {e}")
        status.status = "failed"
        status.error = str(e)
        status.completed_at = datetime.now().isoformat()
        _save_job_status(status)

    return status


async def wait_for_completion(job_id: Optional[str] = None, poll_interval: float = 30.0) -> JobStatus:
    """Wait for a specific job to complete.

    Args:
        job_id: Specific job ID to wait for. If None, waits for most recent job.
        poll_interval: How often to check status (seconds)

    Returns:
        Final job status

    Raises:
        ValueError: If no job exists
        Exception: If job fails
    """
    import asyncio

    # If no job_id specified, get most recent job
    if job_id is None:
        all_jobs = list_all_jobs()
        if not all_jobs:
            raise ValueError("No jobs found")
        job_id = all_jobs[0].job_id
        print(f"Waiting for most recent job: {job_id}")

    status = _load_job_status(job_id)
    if status is None:
        raise ValueError(f"Job not found: {job_id}")

    print(f"Waiting for job {status.job_id} to complete...")
    print(f"Polling every {poll_interval}s")

    while True:
        status = await get_job_status(job_id)

        if status.status == "completed":
            print(f"\n✓ Job completed!")
            print(f"  Checkpoint: {status.checkpoint_path}")
            return status
        elif status.status == "failed":
            raise Exception(f"Job failed: {status.error}")

        # Wait before next poll
        await asyncio.sleep(poll_interval)


async def submit_preprocess_job(
    dataset: str,
    repo_url: str = "https://github.com/darklord1611/repro-mech-interp.git",
    branch: str = "main",
) -> JobStatus:
    """Submit a data preprocessing job asynchronously.

    Args:
        dataset: Dataset name ("gsm8k", "prontoqa", or "prosqa")
        repo_url: GitHub repository URL
        branch: Git branch to checkout

    Returns:
        Job status with function_call_id for tracking
    """
    # Generate job ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = f"preprocess_{dataset}_{timestamp}"

    print(f"Submitting preprocessing job: {job_id}")
    print(f"  Dataset: {dataset}")

    # Ensure app is deployed
    ensure_app_deployed()

    # Get function reference
    preprocess_func = modal.Function.from_name(APP_NAME, "preprocess_data")

    # Spawn async job
    print("Spawning preprocessing job...")
    with modal.enable_output():
        function_call = preprocess_func.spawn(
            dataset=dataset,
            repo_url=repo_url,
            branch=branch,
        )

    # Create and save status
    status = JobStatus(
        job_id=job_id,
        config_hash="preprocess",
        config_name=f"preprocess_{dataset}",
        config_yaml_path="N/A",
        num_gpus=0,
        status="pending",
        function_call_id=function_call.object_id,
        created_at=datetime.now().isoformat(),
    )
    _save_job_status(status)

    print(f"✓ Preprocessing job submitted: {job_id}")
    print(f"  Function Call ID: {function_call.object_id}")

    return status


async def submit_pipeline_job(
    dataset: str,
    cot_config_yaml_path: str,
    coconut_config_yaml_path: str,
    num_gpus: int = 4,
    skip_preprocessing: bool = False,
    skip_cot: bool = False,
    repo_url: str = "https://github.com/darklord1611/repro-mech-interp.git",
    branch: str = "main",
) -> JobStatus:
    """Submit a full training pipeline job asynchronously.

    Args:
        dataset: Dataset name ("gsm8k", "prontoqa", or "prosqa")
        cot_config_yaml_path: Path to CoT config YAML file
        coconut_config_yaml_path: Path to Coconut config YAML file
        num_gpus: Number of GPUs to use
        skip_preprocessing: Skip preprocessing if data already exists
        skip_cot: Skip CoT training if checkpoint already exists
        repo_url: GitHub repository URL
        branch: Git branch to checkout

    Returns:
        Job status with function_call_id for tracking
    """
    # Load configs
    with open(cot_config_yaml_path) as f:
        cot_config = yaml.safe_load(f)
    with open(coconut_config_yaml_path) as f:
        coconut_config = yaml.safe_load(f)

    coconut_name = coconut_config.get("name", "coconut")

    # Compute config hash for coconut config
    config_hash = compute_config_hash(coconut_config, num_gpus)

    # Generate job ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = f"pipeline_{dataset}_{coconut_name}_{config_hash}_{timestamp}"

    print(f"Submitting pipeline job: {job_id}")
    print(f"  Dataset: {dataset}")
    print(f"  CoT Config: {cot_config_yaml_path}")
    print(f"  Coconut Config: {coconut_config_yaml_path}")
    print(f"  GPUs: {num_gpus}")

    # Ensure app is deployed
    ensure_app_deployed()

    # Get function reference
    pipeline_func = modal.Function.from_name(APP_NAME, "train_pipeline")

    # Spawn async job
    print("Spawning pipeline job...")
    with modal.enable_output():
        function_call = pipeline_func.spawn(
            dataset=dataset,
            cot_config_yaml=cot_config,
            coconut_config_yaml=coconut_config,
            num_gpus=num_gpus,
            skip_preprocessing=skip_preprocessing,
            skip_cot=skip_cot,
            repo_url=repo_url,
            branch=branch,
        )

    # Create and save status
    status = JobStatus(
        job_id=job_id,
        config_hash=config_hash,
        config_name=f"pipeline_{dataset}_{coconut_name}",
        config_yaml_path=coconut_config_yaml_path,
        num_gpus=num_gpus,
        status="pending",
        function_call_id=function_call.object_id,
        created_at=datetime.now().isoformat(),
    )
    _save_job_status(status)

    print(f"✓ Pipeline job submitted: {job_id}")
    print(f"  Function Call ID: {function_call.object_id}")

    return status
