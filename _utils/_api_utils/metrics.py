"""Metrics API endpoints."""

import uuid
from typing import Callable, Optional, List, Any, Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field


class VUNMetricsRequest(BaseModel):
    gen_data: str = Field(..., description="Path to generated structures parquet")
    huggingface_dataset: Optional[str] = Field(None, description="HuggingFace dataset for novelty comparison")
    output_csv: Optional[str] = Field(None, description="Output CSV file")
    output_parquet: Optional[str] = Field(None, description="Optional processed parquet with VUN columns")
    sort_metrics_by: Literal["all", "Condition Vector", "both"] = Field("all", description="Grouping mode for VUN metrics output")
    num_workers: int = Field(8, description="Number of parallel workers")
    load_processed_data: Optional[str] = Field(None, description="Pre-processed novelty reference parquet")
    check_comp_novelty: bool = Field(False, description="Whether to compute compositional novelty")
    skip_validity: bool = Field(False, description="Skip validity checks")
    skip_uniqueness: bool = Field(False, description="Skip uniqueness checks")
    bond_length_acceptability_cutoff: float = Field(1.0, description="Cutoff used in bond-length validity checks")
    allow_stated_p1_mismatch: bool = Field(False, description="Allow P1 space-group mismatch during validity checks")


class EHullMetricsRequest(BaseModel):
    post_parquet: str = Field(..., description="Path to postprocessed structures")
    output_parquet: str = Field(..., description="Output parquet with stability metrics")
    mp_data: str = Field("data/mp_computed_structure_entries.json.gz", description="Path to MP computed structure entries data")
    cif_column: str = Field("Generated CIF", description="Name of the CIF column in the parquet input")
    num_workers: int = Field(4, description="Number of parallel workers")
    batch_size: int = Field(100, description="Batch size for MACE E-hull processing")


class XRDMetricsRequest(BaseModel):
    input_parquet: str = Field(..., description="Path to parquet file with generated structures")
    output_parquet: str = Field(..., description="Path to output XRD metrics parquet")
    num_gens: int = Field(0, description="Maximum generations per structure to score")
    num_workers: int = Field(2, description="Number of parallel workers")
    ref_parquet: Optional[str] = Field(None, description="Optional override parquet with reference structures")
    validity_check: Literal["diffcsp", "crystallm", "none"] = Field("diffcsp", description="Validity check mode")
    sort_gens: Literal["rank", "random", "first"] = Field("rank", description="How to choose generations when num_gens is limited")


class PropertyMetricsRequest(BaseModel):
    post_parquet: str = Field(..., description="Path to processed parquet file with VUN metrics")
    parquet_out: Optional[str] = Field(None, description="Optional parquet output with property metrics columns")
    metrics_out: Optional[str] = Field(None, description="Optional CSV output with property metrics summary")
    sort_metrics_by: Literal["all", "Condition Vector", "both"] = Field("all", description="How to group property metrics")
    num_workers: int = Field(8, description="Number of parallel workers")
    property_targets: str = Field("[]", description="JSON list of property columns in condition-vector order")
    property1_normaliser: Literal["power_log", "linear", "None"] = Field("None", description="Normalization method for the first property")
    property2_normaliser: Literal["power_log", "linear", "None"] = Field("None", description="Normalization method for the second property")
    property3_normaliser: Literal["power_log", "linear", "None"] = Field("None", description="Normalization method for the third property")
    max_property1: float = Field(17.89, description="Maximum value for the first property")
    max_property2: float = Field(17.89, description="Maximum value for the second property")
    max_property3: float = Field(17.89, description="Maximum value for the third property")
    min_property1: float = Field(0.0, description="Minimum value for the first property")
    min_property2: float = Field(0.0, description="Minimum value for the second property")
    min_property3: float = Field(0.0, description="Minimum value for the third property")


def register_metrics_routes(
    app: FastAPI,
    create_pending_job: Callable[[str, List[str]], Any],
    run_command: Callable[[str, List[str], Optional[str]], None],
) -> None:
    """Register metrics endpoints on the API app."""

    @app.post("/metrics/vun")
    async def vun_metrics(request: VUNMetricsRequest, background_tasks: BackgroundTasks):
        if not request.huggingface_dataset and not request.load_processed_data:
            raise HTTPException(
                status_code=422,
                detail="Provide at least one novelty reference source: huggingface_dataset or load_processed_data.",
            )

        job_id = str(uuid.uuid4())
        cmd = [
            "python", "-m", "_utils._metrics.VUN_metrics",
            "--input_parquet", request.gen_data,
            "--num_workers", str(request.num_workers),
        ]

        if request.huggingface_dataset:
            cmd.extend(["--huggingface_dataset", request.huggingface_dataset])
        if request.output_csv:
            cmd.extend(["--output_csv", request.output_csv])
        if request.output_parquet:
            cmd.extend(["--output_parquet", request.output_parquet])
        if request.sort_metrics_by:
            cmd.extend(["--sort_metrics_by", request.sort_metrics_by])
        if request.load_processed_data:
            cmd.extend(["--load_processed_data", request.load_processed_data])
        if request.check_comp_novelty:
            cmd.append("--check_comp_novelty")
        if request.skip_validity:
            cmd.append("--skip_validity")
        if request.skip_uniqueness:
            cmd.append("--skip_uniqueness")
        if request.bond_length_acceptability_cutoff != 1.0:
            cmd.extend([
                "--bond_length_acceptability_cutoff",
                str(request.bond_length_acceptability_cutoff),
            ])
        if request.allow_stated_p1_mismatch:
            cmd.append("--allow_stated_p1_mismatch")

        output_target = request.output_parquet or request.output_csv
        background_tasks.add_task(run_command, job_id, cmd, output_target)
        return create_pending_job(job_id, cmd)

    @app.post("/metrics/ehull")
    async def ehull_metrics(request: EHullMetricsRequest, background_tasks: BackgroundTasks):
        job_id = str(uuid.uuid4())
        cmd = [
            "python", "-m", "_utils._metrics.mace_ehull",
            "--post_parquet", request.post_parquet,
            "--output_parquet", request.output_parquet,
            "--mp_data", request.mp_data,
            "--cif_column", request.cif_column,
            "--num_workers", str(request.num_workers),
            "--batch_size", str(request.batch_size),
        ]

        background_tasks.add_task(run_command, job_id, cmd, request.output_parquet)
        return create_pending_job(job_id, cmd)

    @app.post("/metrics/xrd")
    async def xrd_metrics(request: XRDMetricsRequest, background_tasks: BackgroundTasks):
        job_id = str(uuid.uuid4())
        cmd = [
            "python", "-m", "_utils._metrics.XRD_metrics",
            "--input_parquet", request.input_parquet,
            "--num_gens", str(request.num_gens),
            "--num_workers", str(request.num_workers),
            "--output_parquet", request.output_parquet,
            "--validity_check", request.validity_check,
            "--sort_gens", request.sort_gens,
        ]

        if request.ref_parquet:
            cmd.extend(["--ref_parquet", request.ref_parquet])

        background_tasks.add_task(run_command, job_id, cmd, request.output_parquet)
        return create_pending_job(job_id, cmd)

    @app.post("/metrics/property")
    async def property_metrics(request: PropertyMetricsRequest, background_tasks: BackgroundTasks):
        job_id = str(uuid.uuid4())
        cmd = [
            "python", "-m", "_utils._metrics.property_metrics",
            "--post_parquet", request.post_parquet,
            "--sort_metrics_by", request.sort_metrics_by,
            "--num_workers", str(request.num_workers),
            "--property_targets", request.property_targets,
            "--property1_normaliser", request.property1_normaliser,
            "--property2_normaliser", request.property2_normaliser,
            "--property3_normaliser", request.property3_normaliser,
            "--max_property1", str(request.max_property1),
            "--max_property2", str(request.max_property2),
            "--max_property3", str(request.max_property3),
            "--min_property1", str(request.min_property1),
            "--min_property2", str(request.min_property2),
            "--min_property3", str(request.min_property3),
        ]

        if request.parquet_out:
            cmd.extend(["--parquet_out", request.parquet_out])
        if request.metrics_out:
            cmd.extend(["--metrics_out", request.metrics_out])

        output_target = request.parquet_out or request.metrics_out
        background_tasks.add_task(run_command, job_id, cmd, output_target)
        return create_pending_job(job_id, cmd)
