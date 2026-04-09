#!/usr/bin/env python3
"""CLI for async Steerling SFT training jobs on Modal.

Setup:
    1. pip install modal
    2. modal token new
    3. modal secret create github-secret GITHUB_TOKEN=your_token
    4. modal secret create huggingface-secret HF_TOKEN=your_token
    5. modal secret create wandb-secret WANDB_API_KEY=your_key
    6. modal deploy modal_utils/modal_train_sft.py  # deploy once

Usage:
    python modal_utils/run_sft.py submit
    python modal_utils/run_sft.py submit --single-gpu --max-steps 100
    python modal_utils/run_sft.py submit --resume-from /checkpoints/sft_output/checkpoint-1000
    python modal_utils/run_sft.py status
    python modal_utils/run_sft.py status --job-id <id>
    python modal_utils/run_sft.py wait
    python modal_utils/run_sft.py list
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import modal

APP_NAME = "steerling-sft-training"
JOBS_DIR = Path(__file__).parent / "sft_jobs"
JOBS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Job tracking (mirrors modal_services.JobStatus)
# ---------------------------------------------------------------------------

@dataclass
class JobStatus:
    job_id: str
    function_name: str
    function_call_id: Optional[str]
    status: str  # pending | running | completed | failed
    checkpoint_path: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None

    def save(self) -> None:
        (JOBS_DIR / f"{self.job_id}.json").write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def load(job_id: str) -> Optional[JobStatus]:
        p = JOBS_DIR / f"{job_id}.json"
        if not p.exists():
            return None
        return JobStatus(**json.loads(p.read_text()))


def list_all_jobs() -> list[JobStatus]:
    jobs = []
    for p in JOBS_DIR.glob("*.json"):
        try:
            jobs.append(JobStatus(**json.loads(p.read_text())))
        except Exception:
            pass
    return sorted(jobs, key=lambda j: j.created_at or "", reverse=True)


# ---------------------------------------------------------------------------
# Async helpers (same pattern as modal_services.py)
# ---------------------------------------------------------------------------

def _ensure_deployed() -> None:
    try:
        modal.Function.from_name(APP_NAME, "train_sft_ddp")
    except Exception:
        print(f"Deploying '{APP_NAME}'...")
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import importlib
        mod = importlib.import_module("modal_utils.modal_train_sft")
        with modal.enable_output():
            mod.app.deploy()
        print(f"✓ Deployed '{APP_NAME}'")


DATASET_SIZE = 716_987  # darklord1611/tulu-3-sft-mixture-english-clean after filtering


def _compute_max_steps(num_epochs: int, num_gpus: int, batch_size: int, grad_accum: int) -> int:
    import math
    effective_bs = batch_size * num_gpus * grad_accum
    return math.ceil(DATASET_SIZE / effective_bs) * num_epochs


async def prepare_dataset(
    max_seq_len: int = 2048,
    output_dir: str = "/checkpoints/sft_output",
    branch: str = "main",
) -> dict:
    """Run the dataset preparation function (blocking) and return its result."""
    _ensure_deployed()
    func = modal.Function.from_name(APP_NAME, "prepare_dataset")
    print("Preparing dataset (download + tokenise)...")
    with modal.enable_output():
        result = await func.remote.aio(
            max_seq_len=max_seq_len,
            output_dir=output_dir,
            branch=branch,
        )
    print(f"✓ Dataset ready: {result['n_examples']:,} examples at {result['cache_path']}")
    return result


async def submit_job(
    single_gpu: bool = False,
    num_gpus: int = 2,             # must match the gpu= count in modal_train_sft.py
    num_epochs: int = 1,
    max_seq_len: int = 2048,       # 80GB limit: seq=2048 ~58GB; seq=3072 OOMs
    batch_size: int = 1,           # max per-GPU for 80GB at seq=2048
    gradient_accumulation_steps: int = 8,
    lr: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lambda_rec: float = 0.1,
    lambda_indep: float = 0.01,
    save_every: int = 200,
    log_every: int = 50,
    output_dir: str = "/checkpoints/sft_output",
    resume_from: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    branch: str = "main",
) -> JobStatus:
    """Spawn an SFT training job and return immediately."""
    _ensure_deployed()

    # Ensure dataset cache exists on the volume before training starts
    await prepare_dataset(max_seq_len=max_seq_len, output_dir=output_dir, branch=branch)

    _num_gpus = 1 if single_gpu else num_gpus
    max_steps = _compute_max_steps(num_epochs, _num_gpus, batch_size, gradient_accumulation_steps)
    warmup_steps = max(1, max_steps // 100)  # 1% warmup

    print(f"Epochs: {num_epochs}  →  max_steps: {max_steps:,}  warmup: {warmup_steps}")

    func_name = "train_sft_single" if single_gpu else "train_sft_ddp"
    func = modal.Function.from_name(APP_NAME, func_name)

    kwargs: dict = dict(
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr=lr,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lambda_rec=lambda_rec,
        lambda_indep=lambda_indep,
        save_every=save_every,
        log_every=log_every,
        output_dir=output_dir,
        branch=branch,
    )
    if resume_from:
        kwargs["resume_from"] = resume_from
    if wandb_run_name:
        kwargs["wandb_run_name"] = wandb_run_name

    print(f"Spawning {func_name}...")
    with modal.enable_output():
        call = await func.spawn.aio(**kwargs)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job = JobStatus(
        job_id=f"sft_{func_name}_{timestamp}",
        function_name=func_name,
        function_call_id=call.object_id,
        status="pending",
        created_at=datetime.now().isoformat(),
    )
    job.save()
    return job


async def get_job_status(job_id: Optional[str] = None) -> Optional[JobStatus]:
    """Check the current status of a job (non-blocking)."""
    if job_id is None:
        jobs = list_all_jobs()
        if not jobs:
            return None
        job = jobs[0]
        print(f"Checking most recent job: {job.job_id}")
    else:
        job = JobStatus.load(job_id)
        if job is None:
            return None

    if job.status in ("completed", "failed"):
        return job

    try:
        call = modal.FunctionCall.from_id(job.function_call_id)
        try:
            result = call.get(timeout=0)
            job.status = "completed"
            job.checkpoint_path = result.get("checkpoint_path")
            job.completed_at = datetime.now().isoformat()
        except TimeoutError:
            job.status = "running"
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.completed_at = datetime.now().isoformat()

    job.save()
    return job


async def wait_for_completion(
    job_id: Optional[str] = None,
    poll_interval: float = 30.0,
) -> JobStatus:
    """Poll until a job finishes, then return its final status."""
    if job_id is None:
        jobs = list_all_jobs()
        if not jobs:
            raise ValueError("No jobs found")
        job_id = jobs[0].job_id
        print(f"Waiting for most recent job: {job_id}")

    while True:
        job = await get_job_status(job_id)
        if job is None:
            raise ValueError(f"Job not found: {job_id}")
        if job.status == "completed":
            return job
        if job.status == "failed":
            raise RuntimeError(f"Job failed: {job.error}")
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {job.status}...")
        await asyncio.sleep(poll_interval)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Async Steerling SFT training on Modal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # prepare
    pp = sub.add_parser("prepare", help="Download and pre-tokenise dataset (no GPU needed)")
    pp.add_argument("--max-seq-len", type=int, default=2048)
    pp.add_argument("--output-dir", default="/checkpoints/sft_output")
    pp.add_argument("--branch", default="main")

    # submit
    sp = sub.add_parser("submit", help="Spawn a training job (returns immediately)")
    sp.add_argument("--single-gpu", action="store_true", help="Single-GPU run instead of 4×GPU DDP")
    sp.add_argument("--num-gpus", type=int, default=2, help="GPUs allocated (must match modal_train_sft.py gpu= count)")
    sp.add_argument("--num-epochs", type=int, default=1, help="Full passes over the dataset")
    sp.add_argument("--max-seq-len", type=int, default=2048)
    sp.add_argument("--batch-size", type=int, default=1)
    sp.add_argument("--gradient-accumulation-steps", type=int, default=8)
    sp.add_argument("--lr", type=float, default=2e-4)
    sp.add_argument("--lora-r", type=int, default=16)
    sp.add_argument("--lora-alpha", type=int, default=32)
    sp.add_argument("--lora-dropout", type=float, default=0.05)
    sp.add_argument("--lambda-rec", type=float, default=0.1)
    sp.add_argument("--lambda-indep", type=float, default=0.01)
    sp.add_argument("--save-every", type=int, default=200)
    sp.add_argument("--log-every", type=int, default=50)
    sp.add_argument("--output-dir", default="/checkpoints/sft_output")
    sp.add_argument("--resume-from", default=None)
    sp.add_argument("--run-name", default=None, help="W&B run name")
    sp.add_argument("--branch", default="main")
    sp.add_argument("--wait", action="store_true", help="Block until the job finishes")
    sp.add_argument("--poll-interval", type=float, default=30.0)

    # status
    st = sub.add_parser("status", help="Check job status")
    st.add_argument("--job-id", default=None)

    # wait
    wa = sub.add_parser("wait", help="Wait for a job to finish")
    wa.add_argument("--job-id", default=None)
    wa.add_argument("--poll-interval", type=float, default=30.0)

    # list
    sub.add_parser("list", help="List all jobs")

    args = parser.parse_args()

    if args.command == "prepare":
        asyncio.run(prepare_dataset(
            max_seq_len=args.max_seq_len,
            output_dir=args.output_dir,
            branch=args.branch,
        ))

    elif args.command == "submit":
        job = asyncio.run(submit_job(
            single_gpu=args.single_gpu,
            num_gpus=args.num_gpus,
            num_epochs=args.num_epochs,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            lr=args.lr,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lambda_rec=args.lambda_rec,
            lambda_indep=args.lambda_indep,
            save_every=args.save_every,
            log_every=args.log_every,
            output_dir=args.output_dir,
            resume_from=args.resume_from,
            wandb_run_name=args.run_name,
            branch=args.branch,
        ))
        print(f"\n{'='*60}")
        print(f"Job ID:     {job.job_id}")
        print(f"Call ID:    {job.function_call_id}")
        print(f"Function:   {job.function_name}")
        print(f"{'='*60}")
        print(f"\nCheck:  python modal_utils/run_sft.py status --job-id {job.job_id}")
        print(f"Wait:   python modal_utils/run_sft.py wait --job-id {job.job_id}")

        if args.wait:
            print()
            final = asyncio.run(wait_for_completion(job.job_id, args.poll_interval))
            print(f"\n✓ Done. Checkpoint: {final.checkpoint_path}")

    elif args.command == "status":
        job = asyncio.run(get_job_status(args.job_id))
        if job is None:
            print("No job found.")
        else:
            print(f"\nJob ID:     {job.job_id}")
            print(f"Function:   {job.function_name}")
            print(f"Status:     {job.status}")
            print(f"Created:    {job.created_at}")
            if job.completed_at:
                print(f"Completed:  {job.completed_at}")
            if job.checkpoint_path:
                print(f"Checkpoint: {job.checkpoint_path}")
            if job.error:
                print(f"Error:      {job.error}")

    elif args.command == "wait":
        try:
            final = asyncio.run(wait_for_completion(args.job_id, args.poll_interval))
            print(f"\n✓ Done. Checkpoint: {final.checkpoint_path}")
        except (ValueError, RuntimeError) as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "list":
        jobs = list_all_jobs()
        if not jobs:
            print("No jobs found.")
            return
        print(f"\n{'='*90}")
        print(f"{'Job ID':<40}  {'Function':<20}  {'Status':<10}  Created")
        print(f"{'-'*90}")
        for job in jobs:
            created = (job.created_at or "")[:19].replace("T", " ")
            print(f"{job.job_id:<40}  {job.function_name:<20}  {job.status:<10}  {created}")
        print(f"{'='*90}")
        print(f"Total: {len(jobs)} job(s)")


if __name__ == "__main__":
    main()
