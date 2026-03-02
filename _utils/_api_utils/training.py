"""Training API endpoints."""

import uuid
from typing import Callable, Optional, List, Any

from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    config_file: str = Field(..., description="Path to training config JSONC file")
    multi_gpu: Optional[bool] = Field(None, description="Force multi-GPU on/off. If omitted, API auto-selects torchrun when 2+ GPUs are visible")
    nproc_per_node: Optional[int] = Field(None, description="Number of GPUs for multi-GPU training. If omitted, API uses visible GPU count")


def register_training_routes(
    app: FastAPI,
    create_pending_job: Callable[[str, List[str]], Any],
    run_command: Callable[[str, List[str], Optional[str]], None],
) -> None:
    """Register training endpoints on the API app."""

    @app.post("/train")
    async def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
        job_id = str(uuid.uuid4())

        gpu_count = 0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
        except Exception:
            gpu_count = 0

        if request.multi_gpu is None:
            use_multi_gpu = gpu_count >= 2
        else:
            use_multi_gpu = request.multi_gpu and gpu_count >= 2

        if use_multi_gpu:
            chosen_nproc = request.nproc_per_node if request.nproc_per_node is not None else gpu_count
            if gpu_count > 0:
                chosen_nproc = min(chosen_nproc, gpu_count)

            cmd = [
                "torchrun",
                f"--nproc_per_node={chosen_nproc}",
                "-m", "_train",
                "--config", request.config_file,
            ]
            print(f"Training launcher: torchrun (visible_gpus={gpu_count}, nproc_per_node={chosen_nproc})")
        else:
            cmd = [
                "python", "-m", "_train",
                "--config", request.config_file,
            ]
            print(f"Training launcher: python (visible_gpus={gpu_count})")

        background_tasks.add_task(run_command, job_id, cmd, None)
        return create_pending_job(job_id, cmd)
