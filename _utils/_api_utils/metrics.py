"""Metrics API endpoints."""

import uuid
from typing import Callable, Optional, List, Any

from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel, Field


class VUNMetricsRequest(BaseModel):
    gen_data: str = Field(..., description="Path to generated structures parquet")
    huggingface_dataset: str = Field(..., description="HuggingFace dataset for novelty comparison")
    output_csv: str = Field(..., description="Output CSV file")
    num_workers: int = Field(8, description="Number of parallel workers")


class EHullMetricsRequest(BaseModel):
    post_parquet: str = Field(..., description="Path to postprocessed structures")
    output_parquet: str = Field(..., description="Output parquet with stability metrics")
    num_workers: int = Field(4, description="Number of parallel workers")


def register_metrics_routes(
    app: FastAPI,
    create_pending_job: Callable[[str, List[str]], Any],
    run_command: Callable[[str, List[str], Optional[str]], None],
) -> None:
    """Register metrics endpoints on the API app."""

    @app.post("/metrics/vun")
    async def vun_metrics(request: VUNMetricsRequest, background_tasks: BackgroundTasks):
        job_id = str(uuid.uuid4())
        cmd = [
            "python", "-m", "_utils._metrics.VUN_metrics",
            "--input_parquet", request.gen_data,
            "--huggingface_dataset", request.huggingface_dataset,
            "--output_csv", request.output_csv,
            "--num_workers", str(request.num_workers),
        ]

        background_tasks.add_task(run_command, job_id, cmd, request.output_csv)
        return create_pending_job(job_id, cmd)

    @app.post("/metrics/ehull")
    async def ehull_metrics(request: EHullMetricsRequest, background_tasks: BackgroundTasks):
        job_id = str(uuid.uuid4())
        cmd = [
            "python", "-m", "_utils._metrics.mace_ehull",
            "--post_parquet", request.post_parquet,
            "--output_parquet", request.output_parquet,
            "--num_workers", str(request.num_workers),
        ]

        background_tasks.add_task(run_command, job_id, cmd, request.output_parquet)
        return create_pending_job(job_id, cmd)
