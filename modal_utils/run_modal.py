#!/usr/bin/env python3
"""CLI for Modal Coconut training with multi-job support.

Setup:
    1. pip install modal
    2. modal token new
    3. modal secret create github-secret GITHUB_TOKEN=your_token
    4. modal secret create wandb-secret WANDB_API_KEY=your_key

Usage:
    # Submit a job (returns immediately)
    python modal_utils/run_modal.py submit --config args/gsm_coconut.yaml

    # Submit with specific number of GPUs
    python modal_utils/run_modal.py submit --config args/gsm_coconut.yaml --gpus 4

    # List all jobs
    python modal_utils/run_modal.py list

    # Check specific job status
    python modal_utils/run_modal.py status --job-id <job_id>

    # Check most recent job
    python modal_utils/run_modal.py status

    # Wait for specific job
    python modal_utils/run_modal.py wait --job-id <job_id>
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import from modal_utils package
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import asyncio
import yaml
from modal_utils.modal_services import (
    submit_job,
    get_job_status,
    wait_for_completion,
    list_all_jobs,
    submit_preprocess_job,
    submit_pipeline_job,
)


def main():
    parser = argparse.ArgumentParser(
        description="CLI for Modal Coconut training with multi-job support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit training job")
    submit_parser.add_argument("--config", required=True, help="Path to config YAML")
    submit_parser.add_argument("--gpus", type=int, default=4, help="Number of GPUs (default: 4)")
    submit_parser.add_argument("--wait", action="store_true", help="Wait for completion")
    submit_parser.add_argument("--repo-url", default="https://github.com/darklord1611/repro-mech-interp.git")
    submit_parser.add_argument("--branch", default="main")

    # List command
    list_parser = subparsers.add_parser("list", help="List all jobs")
    list_parser.add_argument("--active-only", action="store_true", help="Show only pending/running jobs")

    # Status command
    status_parser = subparsers.add_parser("status", help="Check job status")
    status_parser.add_argument("--job-id", help="Specific job ID (default: most recent)")

    # Wait command
    wait_parser = subparsers.add_parser("wait", help="Wait for job to complete")
    wait_parser.add_argument("--job-id", help="Specific job ID (default: most recent)")
    wait_parser.add_argument("--poll-interval", type=float, default=30.0, help="Poll interval in seconds")

    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess dataset")
    preprocess_parser.add_argument("--dataset", required=True, choices=["gsm8k", "prontoqa", "prosqa"], help="Dataset to preprocess")
    preprocess_parser.add_argument("--repo-url", default="https://github.com/darklord1611/repro-mech-interp.git")
    preprocess_parser.add_argument("--branch", default="main")
    preprocess_parser.add_argument("--wait", action="store_true", help="Wait for completion")

    # Pipeline command (preprocess -> CoT -> Coconut)
    pipeline_parser = subparsers.add_parser("pipeline", help="Run complete training pipeline (preprocess -> CoT -> Coconut)")
    pipeline_parser.add_argument("--dataset", required=True, choices=["gsm8k", "prontoqa", "prosqa"], help="Dataset to use")
    pipeline_parser.add_argument("--cot-config", required=True, help="Path to CoT config YAML")
    pipeline_parser.add_argument("--coconut-config", required=True, help="Path to Coconut config YAML")
    pipeline_parser.add_argument("--gpus", type=int, default=4, help="Number of GPUs (default: 4)")
    pipeline_parser.add_argument("--skip-preprocessing", action="store_true", help="Skip preprocessing step")
    pipeline_parser.add_argument("--skip-cot", action="store_true", help="Skip CoT training step")
    pipeline_parser.add_argument("--repo-url", default="https://github.com/darklord1611/repro-mech-interp.git")
    pipeline_parser.add_argument("--branch", default="main")
    pipeline_parser.add_argument("--wait", action="store_true", help="Wait for completion")

    args = parser.parse_args()

    if args.command == "submit":
        print(f"Submitting job: {args.config} ({args.gpus} GPUs)\n")

        status = asyncio.run(submit_job(
            config_yaml_path=args.config,
            num_gpus=args.gpus,
            repo_url=args.repo_url,
            branch=args.branch,
        ))

        print(f"\n{'='*60}")
        print(f"Job ID: {status.job_id}")
        print(f"Config Hash: {status.config_hash}")
        print(f"Status: {status.status}")
        print(f"Function Call ID: {status.function_call_id}")
        print(f"{'='*60}")

        if args.wait:
            print("\nWaiting for completion...\n")
            final_status = asyncio.run(wait_for_completion(status.job_id))
            print(f"\nCheckpoint: {final_status.checkpoint_path}")

    elif args.command == "list":
        jobs = list_all_jobs()

        if args.active_only:
            jobs = [j for j in jobs if j.status in ["pending", "running"]]

        if not jobs:
            print("No jobs found")
        else:
            print(f"\n{'='*80}")
            print(f"{'Job ID':<50} {'Status':<12} {'Config Hash':<10}")
            print(f"{'-'*80}")
            for job in jobs:
                print(f"{job.job_id:<50} {job.status:<12} {job.config_hash:<10}")
            print(f"{'='*80}")
            print(f"Total: {len(jobs)} job(s)")

    elif args.command == "status":
        status = asyncio.run(get_job_status(args.job_id))

        if status is None:
            if args.job_id:
                print(f"Job not found: {args.job_id}")
            else:
                print("No jobs found")
        else:
            print(f"\n{'='*60}")
            print(f"Job ID: {status.job_id}")
            print(f"Config Hash: {status.config_hash}")
            print(f"Config: {status.config_yaml_path}")
            print(f"GPUs: {status.num_gpus}")
            print(f"Status: {status.status}")
            print(f"Created: {status.created_at}")
            if status.completed_at:
                print(f"Completed: {status.completed_at}")
            if status.checkpoint_path:
                print(f"Checkpoint: {status.checkpoint_path}")
            if status.error:
                print(f"Error: {status.error}")
            print(f"{'='*60}")

    elif args.command == "wait":
        try:
            final_status = asyncio.run(wait_for_completion(args.job_id, args.poll_interval))
            print(f"\nCheckpoint: {final_status.checkpoint_path}")
        except ValueError as e:
            print(f"Error: {e}")
            exit(1)
        except Exception as e:
            print(f"Job failed: {e}")
            exit(1)

    elif args.command == "preprocess":
        print(f"Submitting preprocessing job for {args.dataset}\n")

        status = asyncio.run(submit_preprocess_job(
            dataset=args.dataset,
            repo_url=args.repo_url,
            branch=args.branch,
        ))

        print(f"\n{'='*60}")
        print(f"Job ID: {status.job_id}")
        print(f"Dataset: {args.dataset}")
        print(f"Status: {status.status}")
        print(f"Function Call ID: {status.function_call_id}")
        print(f"{'='*60}")

        if args.wait:
            print("\nWaiting for completion...\n")
            final_status = asyncio.run(wait_for_completion(status.job_id))
            print(f"\nData path: {final_status.checkpoint_path}")

    elif args.command == "pipeline":
        print(f"Submitting full training pipeline for {args.dataset}\n")

        # Load config files
        with open(args.cot_config) as f:
            cot_config = yaml.safe_load(f)
        with open(args.coconut_config) as f:
            coconut_config = yaml.safe_load(f)

        status = asyncio.run(submit_pipeline_job(
            dataset=args.dataset,
            cot_config_yaml_path=args.cot_config,
            coconut_config_yaml_path=args.coconut_config,
            num_gpus=args.gpus,
            skip_preprocessing=args.skip_preprocessing,
            skip_cot=args.skip_cot,
            repo_url=args.repo_url,
            branch=args.branch,
        ))

        print(f"\n{'='*60}")
        print(f"Job ID: {status.job_id}")
        print(f"Dataset: {args.dataset}")
        print(f"Pipeline: {'[skip] ' if args.skip_preprocessing else ''}preprocess -> {'[skip] ' if args.skip_cot else ''}CoT -> Coconut")
        print(f"Status: {status.status}")
        print(f"Function Call ID: {status.function_call_id}")
        print(f"{'='*60}")

        if args.wait:
            print("\nWaiting for completion...\n")
            final_status = asyncio.run(wait_for_completion(status.job_id))
            print(f"\nCoconut checkpoint: {final_status.checkpoint_path}")


if __name__ == "__main__":
    main()
