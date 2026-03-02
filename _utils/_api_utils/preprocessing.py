"""Preprocessing API endpoints."""

import uuid
from typing import Callable, Optional, List, Literal, Any

from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel, Field


class DeduplicateRequest(BaseModel):
    input_parquet: str = Field(..., description="Path to input parquet file")
    output_parquet: str = Field(..., description="Path to output parquet file")
    property_columns: Optional[str] = Field(None, description="JSON list of property columns, e.g., \"['Bandgap (eV)', 'Density (g/cm^3)']\"")
    filter_na_columns: Optional[str] = Field(None, description="JSON list of columns to filter NaN values")
    filter_zero_columns: Optional[str] = Field(None, description="JSON list of columns to filter zero values")
    filter_negative_columns: Optional[str] = Field(None, description="JSON list of columns to filter negative values")


class CleaningRequest(BaseModel):
    input_parquet: str = Field(..., description="Path to input parquet file")
    output_parquet: str = Field(..., description="Path to output parquet file")
    num_workers: int = Field(8, description="Number of parallel workers")
    property_columns: Optional[str] = Field(None, description="JSON list of property columns")
    property1_normaliser: Optional[Literal["linear", "power_log", "signed_log", "log10", "None"]] = None
    property2_normaliser: Optional[Literal["linear", "power_log", "signed_log", "log10", "None"]] = None


class SaveDatasetRequest(BaseModel):
    input_parquet: str = Field(..., description="Path to processed parquet file")
    output_parquet: str = Field(..., description="Dataset name for HuggingFace")
    test_size: float = Field(0.1, description="Test split size")
    valid_size: float = Field(0.1, description="Validation split size")
    HF_username: str = Field(..., description="HuggingFace username")
    save_hub: bool = Field(True, description="Save to HuggingFace Hub")
    save_local: bool = Field(False, description="Save locally")
    duplicates: bool = Field(False, description="Split by Material ID to prevent leakage")


class XRDPreprocessRequest(BaseModel):
    input_data: str = Field(..., description="Path to input XRD file (.csv, .xy, .dat, .txt)")
    output_csv: str = Field(..., description="Path to output CrystaLLM-formatted peak CSV")
    xrd_wavelength: float = Field(1.54056, description="Original wavelength (Angstrom)")


class CalcTheorXRDRequest(BaseModel):
    input_parquet: str = Field(..., description="Path to input parquet containing CIFs")
    output_parquet: str = Field(..., description="Path to output parquet with theoretical XRD condition vectors")
    num_workers: int = Field(4, description="Number of parallel workers")
    column_name: str = Field("CIF", description="Name of the column containing CIF strings")


class CifsZipToParquetRequest(BaseModel):
    input_tarballs: List[str] = Field(..., description="Path(s) to input CIF tar.gz file(s)")
    output_parquet: str = Field(..., description="Path to output parquet file")
    database_name: str = Field(..., description="Database label to write in output parquet")
    num_workers: int = Field(8, description="Number of parallel workers for CIF processing")


def register_preprocessing_routes(
    app: FastAPI,
    create_pending_job: Callable[[str, List[str]], Any],
    run_command: Callable[[str, List[str], Optional[str]], None],
) -> None:
    """Register preprocessing endpoints on the API app."""

    @app.post("/preprocessing/deduplicate")
    async def deduplicate(request: DeduplicateRequest, background_tasks: BackgroundTasks):
        job_id = str(uuid.uuid4())
        cmd = [
            "python", "-m", "_utils._preprocessing._deduplicate",
            "--input_parquet", request.input_parquet,
            "--output_parquet", request.output_parquet,
        ]

        if request.property_columns:
            cmd.extend(["--property_columns", request.property_columns])
        if request.filter_na_columns:
            cmd.extend(["--filter_na_columns", request.filter_na_columns])
        if request.filter_zero_columns:
            cmd.extend(["--filter_zero_columns", request.filter_zero_columns])
        if request.filter_negative_columns:
            cmd.extend(["--filter_negative_columns", request.filter_negative_columns])

        background_tasks.add_task(run_command, job_id, cmd, request.output_parquet)
        return create_pending_job(job_id, cmd)

    @app.post("/preprocessing/clean")
    async def clean_data(request: CleaningRequest, background_tasks: BackgroundTasks):
        job_id = str(uuid.uuid4())
        cmd = [
            "python", "-m", "_utils._preprocessing._cleaning",
            "--input_parquet", request.input_parquet,
            "--output_parquet", request.output_parquet,
            "--num_workers", str(request.num_workers),
        ]

        if request.property_columns:
            cmd.extend(["--property_columns", request.property_columns])
        if request.property1_normaliser:
            cmd.extend(["--property1_normaliser", request.property1_normaliser])
        if request.property2_normaliser:
            cmd.extend(["--property2_normaliser", request.property2_normaliser])

        background_tasks.add_task(run_command, job_id, cmd, request.output_parquet)
        return create_pending_job(job_id, cmd)

    @app.post("/preprocessing/save-dataset")
    async def save_dataset(request: SaveDatasetRequest, background_tasks: BackgroundTasks):
        job_id = str(uuid.uuid4())
        cmd = [
            "python", "-m", "_utils._preprocessing._save_dataset_to_HF",
            "--input_parquet", request.input_parquet,
            "--output_parquet", request.output_parquet,
            "--test_size", str(request.test_size),
            "--valid_size", str(request.valid_size),
            "--HF_username", request.HF_username,
        ]

        if request.save_hub:
            cmd.append("--save_hub")
        if request.save_local:
            cmd.append("--save_local")
        if request.duplicates:
            cmd.append("--duplicates")

        background_tasks.add_task(run_command, job_id, cmd, None)
        return create_pending_job(job_id, cmd)

    @app.post("/preprocessing/process-exp-xrd")
    async def preprocess_exp_xrd(request: XRDPreprocessRequest, background_tasks: BackgroundTasks):
        job_id = str(uuid.uuid4())
        cmd = [
            "python", "-m", "_utils._preprocessing._process_exp_XRD_inputs",
            "--input_data", request.input_data,
            "--xrd_wavelength", str(request.xrd_wavelength),
            "--output_csv", request.output_csv,
        ]

        background_tasks.add_task(run_command, job_id, cmd, request.output_csv)
        return create_pending_job(job_id, cmd)

    @app.post("/preprocessing/calc-theor-xrd")
    async def calc_theor_xrd(request: CalcTheorXRDRequest, background_tasks: BackgroundTasks):
        job_id = str(uuid.uuid4())
        cmd = [
            "python", "-m", "_utils._preprocessing._calculate_theor_XRD",
            "--input_parquet", request.input_parquet,
            "--output_parquet", request.output_parquet,
            "--num_workers", str(request.num_workers),
            "--column_name", request.column_name,
        ]

        background_tasks.add_task(run_command, job_id, cmd, request.output_parquet)
        return create_pending_job(job_id, cmd)

    @app.post("/preprocessing/cifs-zip-to-parquet")
    async def cifs_zip_to_parquet(request: CifsZipToParquetRequest, background_tasks: BackgroundTasks):
        job_id = str(uuid.uuid4())
        cmd = ["python", "-m", "_utils._preprocessing._cifs_zip_to_parquet", "--input_tarballs"]
        cmd.extend(request.input_tarballs)
        cmd.extend([
            "--output_parquet", request.output_parquet,
            "--database_name", request.database_name,
            "--num_workers", str(request.num_workers),
        ])

        background_tasks.add_task(run_command, job_id, cmd, request.output_parquet)
        return create_pending_job(job_id, cmd)
