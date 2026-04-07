"""Sequential training pipeline: trains multiple models one after another."""

import subprocess
import datetime
import os
import sys
import commentjson

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Global Hardware Settings
NUM_GPUS = 2
GPU_TO_USE = 1  # Only used if NUM_GPUS == 1

TRAINING_JOBS = [
    # {
    #     "name": "ResidualGraph_ft_ln_2",
    #     "train_config": "_config_files/training/unconditional/graph_mask/residual-graph_ft_ln.jsonc",
    #     "output_dir": "model_ckpts/ResidualGraph_ft_ln_2/"
    # },
    # {
    #     "name": "ResidualGraph_ft_am",
    #     "train_config": "_config_files/training/unconditional/graph_mask/residual-graph_ft_am.jsonc",
    #     "output_dir": "model_ckpts/ResidualGraph_ft_am/"
    # },
    {
        "name": "alex_mp_20_graph_teacher",
        "train_config": "_config_files/training/unconditional/lupi_kd/alex-mp-20-teacher.jsonc",
        "output_dir": "model_ckpts/alex_mp_20_graph_teacher/"
    }
]


def resolve_path(p):
    """Converts relative paths to absolute paths based on BASE_DIR."""
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.join(BASE_DIR, p)


def resolve_cfg_paths(cfg, keys):
    """Resolve a set of config keys in-place to absolute paths."""
    for k in keys:
        if k in cfg and cfg[k]:
            cfg[k] = resolve_path(cfg[k])


def validate_jobs(jobs):
    """Check that all required config files exist before starting."""
    all_valid = True
    for job in jobs:
        name = job.get("name", "unknown")
        path = resolve_path(job.get("train_config"))
        
        if not path:
            print(f"ERROR [{name}]: Missing 'train_config' path")
            all_valid = False
        elif not os.path.exists(path):
            print(f"ERROR [{name}]: Config file not found: '{path}'")
            all_valid = False
            
    return all_valid


def _preflight_check():
    """Ensure the main training script exists."""
    train_script = resolve_path("_train.py")
    if not os.path.exists(train_script):
        print(f"ERROR: Missing '_train.py' at {train_script}")
        return False
    return True


def run_cmd(cmd, step_name, log_dir, env_override=None):
    """Run command with stdout/stderr logged to files."""
    os.makedirs(log_dir, exist_ok=True)
    out_log = f"{log_dir}/{step_name}_stdout.log"
    err_log = f"{log_dir}/{step_name}_stderr.log"
    
    # Merge env if provided
    proc_env = os.environ.copy()
    if env_override:
        proc_env.update(env_override)
    
    try:
        with open(out_log, "w") as out_f, open(err_log, "w") as err_f:
            subprocess.check_call(cmd, stdout=out_f, stderr=err_f, env=proc_env, cwd=BASE_DIR)
        print(f"    {step_name} done. Logs: {out_log}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"    {step_name} FAILED (exit code {e.returncode}). See: {err_log}")
        return False


def run_training(job, timestamp):
    """Run a single training job."""
    name = job["name"]
    log_dir = f"__logs__/_{timestamp}/{name}"
    
    print(f"\nStarting job: {name}")

    # Ensure output directory exists
    job_output_dir = resolve_path(job["output_dir"])
    os.makedirs(job_output_dir, exist_ok=True)

    # Setup environment for GPU visibility
    env_for_processes = None
    if NUM_GPUS == 1 and GPU_TO_USE is not None:
        env_for_processes = {"CUDA_VISIBLE_DEVICES": str(GPU_TO_USE)}

    # Load and process the config file
    config_path = resolve_path(job["train_config"])
    with open(config_path, "r") as f:
        train_cfg = commentjson.load(f)

    # Force the output directory to match the job definition
    train_cfg["output_dir"] = job_output_dir

    # Resolve internal paths so the training script finds them easily
    resolve_cfg_paths(
        train_cfg,
        ["output_dir", "pretrained_tokenizer_dir", "pretrained_model_dir"]
    )

    # Write a temporary config file to the output dir.
    # This preserves the exact config used for this run.
    tmp_config_path = os.path.join(job_output_dir, os.path.basename(config_path))
    with open(tmp_config_path, "w") as f:
        commentjson.dump(train_cfg, f, indent=2)

    print(f"  Step: Training model...")
    
    # Construct command based on GPU count
    if NUM_GPUS > 1:
        cmd = [
            "torchrun", f"--nproc_per_node={NUM_GPUS}", "_train.py",
            "--config", tmp_config_path
        ]
    else:
        cmd = [
            sys.executable, "_train.py",
            "--config", tmp_config_path
        ]

    success = run_cmd(cmd, "training", log_dir, env_override=env_for_processes)
    
    if success:
        print(f"  Job '{name}' completed successfully.")
    else:
        print(f"  Job '{name}' FAILED.")
        
    return success


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Sequential Training Pipeline started at {timestamp}")
    print(f"Scheduled {len(TRAINING_JOBS)} job(s)")

    if not _preflight_check():
        return

    print("\nValidating job configurations...")
    if not validate_jobs(TRAINING_JOBS):
        print("\nValidation FAILED. Please fix the missing files above before running.")
        return
    print("All configurations validated.")

    results = []
    for i, job in enumerate(TRAINING_JOBS, 1):
        print(f"\nJob [{i}/{len(TRAINING_JOBS)}]")
        success = run_training(job, timestamp)
        results.append((job["name"], success))

    # Summary
    print("\nPipeline Summary")
    for name, success in results:
        status = "SUCCESS" if success else "FAILED"
        print(f"  {name}: {status}")
    
    failed = sum(1 for _, s in results if not s)
    print(f"\nCompleted: {len(results) - failed}/{len(results)} jobs succeeded.")
    print(f"Logs can be found in __logs__/_{timestamp}/")


if __name__ == "__main__":
    main()