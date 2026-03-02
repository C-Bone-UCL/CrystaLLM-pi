"""FastAPI wrapper for CrystaLLM-pi CLI utilities, endpoint groups are from `_utils._api_utils` modules.
"""

import os
import signal
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, List

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError:
    raise ImportError("This modules is meant to be run in the containerized API environment where FastAPI and Pydantic are available. If you want to run it locally, please install the required dependencies first.")

from _utils._api_utils.preprocessing import register_preprocessing_routes
from _utils._api_utils.training import register_training_routes
from _utils._api_utils.generation import register_generation_routes
from _utils._api_utils.metrics import register_metrics_routes
from _utils._api_utils.jobs import register_jobs_routes


app = FastAPI(
    title="CrystaLLM-pi API",
    description="REST API for CrystaLLM-pi crystallography utilities",
    version="1.0.0",
)


_JOB_STORE_DIR = Path(os.environ.get("API_JOB_STORE_DIR", "/tmp/crystallm_jobs"))
_JOB_STORE_DIR.mkdir(parents=True, exist_ok=True)
_JOB_STORE_LOCK = threading.Lock()
_PROCESS_LOCK = threading.Lock()
_ACTIVE_PROCESSES = {}


class JobStatus(BaseModel):
    job_id: str
    status: Literal["pending", "running", "completed", "failed"]
    command: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    output_file: Optional[str] = None
    log_file: Optional[str] = None


class _JobsCompat(dict):
    """Compat object so tests can call `jobs.clear()` before each run."""

    def clear(self):
        with _JOB_STORE_LOCK:
            for pattern in ("*.json", "*.json.tmp"):
                for job_file in _JOB_STORE_DIR.glob(pattern):
                    try:
                        job_file.unlink()
                    except Exception:
                        continue
        super().clear()


jobs = _JobsCompat()


def _job_file_path(job_id: str) -> Path:
    return _JOB_STORE_DIR / f"{job_id}.json"


def _save_job(job: JobStatus):
    payload = job.model_dump_json(indent=2)
    with _JOB_STORE_LOCK:
        tmp_path = _job_file_path(job.job_id).with_suffix(".json.tmp")
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(_job_file_path(job.job_id))
    jobs[job.job_id] = job.model_dump()


def _load_job(job_id: str) -> Optional[JobStatus]:
    job_path = _job_file_path(job_id)
    if not job_path.exists():
        return None
    try:
        raw = job_path.read_text(encoding="utf-8")
        return JobStatus.model_validate_json(raw)
    except Exception:
        return None


def _update_job(job_id: str, **updates) -> Optional[JobStatus]:
    with _JOB_STORE_LOCK:
        job_path = _job_file_path(job_id)
        if not job_path.exists():
            return None
        try:
            current = JobStatus.model_validate_json(job_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        merged = current.model_copy(update=updates)
        tmp_path = job_path.with_suffix(".json.tmp")
        tmp_path.write_text(merged.model_dump_json(indent=2), encoding="utf-8")
        tmp_path.replace(job_path)
        jobs[job_id] = merged.model_dump()
        return merged


def _list_jobs() -> List[JobStatus]:
    stored_jobs: List[JobStatus] = []
    with _JOB_STORE_LOCK:
        for job_path in sorted(_JOB_STORE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                stored_jobs.append(JobStatus.model_validate_json(job_path.read_text(encoding="utf-8")))
            except Exception:
                continue

    dict.clear(jobs)
    for item in stored_jobs:
        jobs[item.job_id] = item.model_dump()
    return stored_jobs


def _create_pending_job(job_id: str, cmd: List[str]) -> JobStatus:
    job = JobStatus(
        job_id=job_id,
        status="pending",
        command=" ".join(cmd),
        started_at=datetime.now().isoformat(),
    )
    _save_job(job)
    return job


def run_command(job_id: str, cmd: List[str], output_file: Optional[str]):
    """Execute command and update job status."""
    current_job = _load_job(job_id)
    if current_job and current_job.status == "failed" and (current_job.error or "").startswith("Cancelled by user"):
        return

    _update_job(job_id, status="running")
    log_file = Path(f"/tmp/{job_id}.log")

    try:
        log_dir_value = os.environ.get("API_JOB_LOG_DIR", "/app/outputs/api_job_logs")
        try:
            log_dir = Path(log_dir_value)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"{job_id}.log"
        except Exception:
            fallback_dir = Path("/tmp/api_job_logs")
            fallback_dir.mkdir(parents=True, exist_ok=True)
            log_file = fallback_dir / f"{job_id}.log"

        _update_job(job_id, log_file=str(log_file))

        command_env = os.environ.copy()
        cache_root = Path(os.environ.get("API_RUNTIME_ROOT", "/tmp/crystallm_runtime"))
        try:
            (cache_root / ".tmp").mkdir(parents=True, exist_ok=True)
        except Exception:
            cache_root = Path("/tmp/crystallm_api")
            (cache_root / ".tmp").mkdir(parents=True, exist_ok=True)

        command_env["HF_HOME"] = str(cache_root / ".cache" / "huggingface")
        command_env["HF_HUB_CACHE"] = str(cache_root / ".cache" / "huggingface" / "hub")
        command_env["HF_ASSETS_CACHE"] = str(cache_root / ".cache" / "huggingface" / "assets")
        command_env["XDG_CACHE_HOME"] = str(cache_root / ".cache")
        command_env["TMPDIR"] = str(cache_root / ".tmp")
        command_env["HOME"] = str(cache_root)
        command_env["TORCH_HOME"] = str(cache_root / ".cache" / "torch")
        command_env["PYTHONDONTWRITEBYTECODE"] = "1"

        for path_str in [
            command_env["HF_HOME"],
            command_env["HF_HUB_CACHE"],
            command_env["HF_ASSETS_CACHE"],
            command_env["XDG_CACHE_HOME"],
            command_env["TMPDIR"],
            command_env["TORCH_HOME"],
            command_env["HOME"],
        ]:
            Path(path_str).mkdir(parents=True, exist_ok=True)

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="/app",
            env=command_env,
            start_new_session=True,
        )

        with _PROCESS_LOCK:
            _ACTIVE_PROCESSES[job_id] = process

        stdout, stderr = process.communicate()
        return_code = process.returncode

        if return_code != 0:
            raise subprocess.CalledProcessError(
                returncode=return_code,
                cmd=cmd,
                output=stdout,
                stderr=stderr,
            )

        with open(log_file, "w", encoding="utf-8") as handle:
            handle.write(f"Command: {' '.join(cmd)}\n")
            handle.write("\n=== STDOUT ===\n")
            handle.write(stdout or "")
            handle.write("\n=== STDERR ===\n")
            handle.write(stderr or "")

        _update_job(
            job_id,
            status="completed",
            completed_at=datetime.now().isoformat(),
            output_file=output_file,
        )

    except subprocess.CalledProcessError as error:
        _update_job(job_id, status="failed", completed_at=datetime.now().isoformat())
        with open(log_file, "w", encoding="utf-8") as handle:
            handle.write(f"Command: {' '.join(cmd)}\n")
            handle.write(f"Return code: {error.returncode}\n")
            handle.write("\n=== STDOUT ===\n")
            handle.write(error.stdout or "")
            handle.write("\n=== STDERR ===\n")
            handle.write(error.stderr or "")
        _update_job(
            job_id,
            error=(
                f"Command failed (code {error.returncode}). See {log_file}.\n"
                f"Stderr: {error.stderr}\nStdout: {error.stdout}"
            ),
        )
    except Exception as error:
        _update_job(job_id, status="failed", completed_at=datetime.now().isoformat())
        with open(log_file, "w", encoding="utf-8") as handle:
            handle.write(f"Command: {' '.join(cmd)}\n")
            handle.write("\n=== INTERNAL ERROR ===\n")
            handle.write(str(error))
        _update_job(job_id, error=f"{error}. See {log_file}.")
    finally:
        with _PROCESS_LOCK:
            _ACTIVE_PROCESSES.pop(job_id, None)


def _cancel_job(job_id: str):
    """Cancel a running or pending job by id."""
    job = _load_job(job_id)
    if job is None:
        return False, "Job not found"

    if job.status in ("completed", "failed"):
        return False, f"Job already {job.status}"

    with _PROCESS_LOCK:
        process = _ACTIVE_PROCESSES.get(job_id)

    if process is None:
        _update_job(
            job_id,
            status="failed",
            completed_at=datetime.now().isoformat(),
            error="Cancelled by user",
        )
        return True, "Pending job cancelled"

    try:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception:
            process.terminate()

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except Exception:
                process.kill()

        _update_job(
            job_id,
            status="failed",
            completed_at=datetime.now().isoformat(),
            error="Cancelled by user",
        )
        return True, "Job cancelled"
    except Exception as exc:
        return False, f"Failed to cancel job: {exc}"


register_preprocessing_routes(app, _create_pending_job, run_command)
register_training_routes(app, _create_pending_job, run_command)
register_generation_routes(app, _create_pending_job, run_command)
register_metrics_routes(app, _create_pending_job, run_command)
register_jobs_routes(app, _load_job, _list_jobs, _cancel_job)


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "CrystaLLM-pi API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "health": ["/healthz"],
            "preprocessing": [
                "/preprocessing/deduplicate",
                "/preprocessing/calc-theor-xrd",
                "/preprocessing/process-exp-xrd",
                "/preprocessing/clean",
                "/preprocessing/save-dataset",
                "/preprocessing/cifs-zip-to-parquet",
            ],
            "training": ["/train"],
            "generation": [
                "/generate/direct",
                "/generate/make-prompts",
                "/generate/cifs",
                "/generate/evaluate-cifs",
                "/generate/postprocess",
            ],
            "metrics": [
                "/metrics/vun",
                "/metrics/ehull",
            ],
            "jobs": [
                "/jobs",
                "/jobs/{job_id}",
                "/jobs/{job_id}/cancel",
            ],
        },
    }


@app.get("/healthz")
async def healthz():
    """Lightweight health check endpoint."""
    return {
        "status": "ok",
        "service": "CrystaLLM-pi API",
        "time": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
