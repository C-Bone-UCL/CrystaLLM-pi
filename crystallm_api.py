"""
FastAPI wrapper for CrystaLLM-2.0 CLI utilities
Provides REST API endpoints for all command-line tools
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
import subprocess
import json
import uuid
from pathlib import Path
from datetime import datetime

app = FastAPI(
    title="CrystaLLM-2.0 API",
    description="REST API for CrystaLLM-2.0 crystallography utilities",
    version="1.0.0"
)

# Storage for async job tracking
jobs = {}

class JobStatus(BaseModel):
    job_id: str
    status: Literal["pending", "running", "completed", "failed"]
    command: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    output_file: Optional[str] = None


# ==================== DATA PREPROCESSING ====================

class DeduplicateRequest(BaseModel):
    input_parquet: str = Field(..., description="Path to input parquet file")
    output_parquet: str = Field(..., description="Path to output parquet file")
    property_columns: Optional[str] = Field(None, description="JSON list of property columns, e.g., \"['Bandgap (eV)', 'Density (g/cm^3)']\"")
    filter_na_columns: Optional[str] = Field(None, description="JSON list of columns to filter NaN values")
    filter_zero_columns: Optional[str] = Field(None, description="JSON list of columns to filter zero values")
    filter_negative_columns: Optional[str] = Field(None, description="JSON list of columns to filter negative values")


@app.post("/preprocessing/deduplicate", response_model=JobStatus)
async def deduplicate(request: DeduplicateRequest, background_tasks: BackgroundTasks):
    """Deduplicate and filter crystallographic data"""
    job_id = str(uuid.uuid4())
    
    cmd = [
        "python", "-m", "_utils._preprocessing._deduplicate",
        "--input_parquet", request.input_parquet,
        "--output_parquet", request.output_parquet
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
    
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        command=" ".join(cmd),
        started_at=datetime.now().isoformat()
    )
    
    return jobs[job_id]


class CleaningRequest(BaseModel):
    input_parquet: str = Field(..., description="Path to input parquet file")
    output_parquet: str = Field(..., description="Path to output parquet file")
    num_workers: int = Field(8, description="Number of parallel workers")
    property_columns: Optional[str] = Field(None, description="JSON list of property columns")
    property1_normaliser: Optional[Literal["linear", "power_log", "signed_log", "log10", "None"]] = None
    property2_normaliser: Optional[Literal["linear", "power_log", "signed_log", "log10", "None"]] = None


@app.post("/preprocessing/clean", response_model=JobStatus)
async def clean_data(request: CleaningRequest, background_tasks: BackgroundTasks):
    """Clean and normalize CIF data"""
    job_id = str(uuid.uuid4())
    
    cmd = [
        "python", "-m", "_utils._preprocessing._cleaning",
        "--input_parquet", request.input_parquet,
        "--output_parquet", request.output_parquet,
        "--num_workers", str(request.num_workers)
    ]
    
    if request.property_columns:
        cmd.extend(["--property_columns", request.property_columns])
    if request.property1_normaliser:
        cmd.extend(["--property1_normaliser", request.property1_normaliser])
    if request.property2_normaliser:
        cmd.extend(["--property2_normaliser", request.property2_normaliser])
    
    background_tasks.add_task(run_command, job_id, cmd, request.output_parquet)
    
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        command=" ".join(cmd),
        started_at=datetime.now().isoformat()
    )
    
    return jobs[job_id]


class SaveDatasetRequest(BaseModel):
    input_parquet: str = Field(..., description="Path to processed parquet file")
    output_parquet: str = Field(..., description="Dataset name for HuggingFace")
    test_size: float = Field(0.1, description="Test split size")
    valid_size: float = Field(0.1, description="Validation split size")
    HF_username: str = Field(..., description="HuggingFace username")
    save_hub: bool = Field(True, description="Save to HuggingFace Hub")
    save_local: bool = Field(False, description="Save locally")
    duplicates: bool = Field(False, description="Split by Material ID to prevent leakage")


@app.post("/preprocessing/save-dataset", response_model=JobStatus)
async def save_dataset(request: SaveDatasetRequest, background_tasks: BackgroundTasks):
    """Save dataset to HuggingFace with train/val/test splits"""
    job_id = str(uuid.uuid4())
    
    cmd = [
        "python", "-m", "_utils._preprocessing._save_dataset_to_HF",
        "--input_parquet", request.input_parquet,
        "--output_parquet", request.output_parquet,
        "--test_size", str(request.test_size),
        "--valid_size", str(request.valid_size),
        "--HF_username", request.HF_username
    ]
    
    if request.save_hub:
        cmd.append("--save_hub")
    if request.save_local:
        cmd.append("--save_local")
    if request.duplicates:
        cmd.append("--duplicates")
    
    background_tasks.add_task(run_command, job_id, cmd, None)
    
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        command=" ".join(cmd),
        started_at=datetime.now().isoformat()
    )
    
    return jobs[job_id]


# ==================== TRAINING ====================

class TrainRequest(BaseModel):
    config_file: str = Field(..., description="Path to training config JSONC file")
    multi_gpu: bool = Field(False, description="Use multi-GPU training with torchrun")
    nproc_per_node: int = Field(2, description="Number of GPUs for multi-GPU training")


@app.post("/train", response_model=JobStatus)
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    """Train or fine-tune a CrystaLLM model"""
    job_id = str(uuid.uuid4())
    
    if request.multi_gpu:
        cmd = [
            "torchrun",
            f"--nproc_per_node={request.nproc_per_node}",
            "-m", "_train",
            "--config", request.config_file
        ]
    else:
        cmd = [
            "python", "-m", "_train",
            "--config", request.config_file
        ]
    
    background_tasks.add_task(run_command, job_id, cmd, None)
    
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        command=" ".join(cmd),
        started_at=datetime.now().isoformat()
    )
    
    return jobs[job_id]


# ==================== GENERATION ====================

class DirectGenerationRequest(BaseModel):
    hf_model_path: str = Field(..., description="HuggingFace model path (e.g., c-bone/CrystaLLM-2.0_base)")
    output_parquet: str = Field(..., description="Output parquet file path")
    manual: bool = Field(True, description="Manual mode (specify compositions)")
    compositions: Optional[str] = Field(None, description="Comma-separated compositions (e.g., 'LiFePO4,TiO2')")
    condition_lists: Optional[List[str]] = Field(None, description="List of condition values (normalized 0-1)")
    input_parquet: Optional[str] = Field(None, description="Input prompts parquet (alternative to manual)")
    level: Optional[Literal["level_1", "level_2", "level_3", "level_4"]] = Field("level_2", description="Prompt detail level")


@app.post("/generate/direct", response_model=JobStatus)
async def generate_direct(request: DirectGenerationRequest, background_tasks: BackgroundTasks):
    """Direct generation using pre-trained HuggingFace models"""
    job_id = str(uuid.uuid4())
    
    cmd = [
        "python", "-m", "_load_and_generate",
        "--hf_model_path", request.hf_model_path,
        "--output_parquet", request.output_parquet
    ]
    
    if request.manual:
        cmd.append("--manual")
        if request.compositions:
            cmd.extend(["--compositions", request.compositions])
        if request.condition_lists:
            cmd.append("--condition_lists")
            cmd.extend(request.condition_lists)
    elif request.input_parquet:
        cmd.extend(["--input_parquet", request.input_parquet])
    
    background_tasks.add_task(run_command, job_id, cmd, request.output_parquet)
    
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        command=" ".join(cmd),
        started_at=datetime.now().isoformat()
    )
    
    return jobs[job_id]


class MakePromptsRequest(BaseModel):
    output_parquet: str = Field(..., description="Output parquet file")
    manual: bool = Field(True, description="Manual mode")
    compositions: Optional[str] = Field(None, description="Comma-separated compositions")
    condition_lists: Optional[List[str]] = Field(None, description="Condition value lists")
    level: Literal["level_1", "level_2", "level_3", "level_4"] = Field("level_2", description="Prompt level")
    spacegroups: Optional[str] = Field(None, description="Space groups (for level_4)")
    automatic: bool = Field(False, description="Automatic mode (extract from dataset)")
    HF_dataset: Optional[str] = Field(None, description="HuggingFace dataset name")
    split: Optional[str] = Field("test", description="Dataset split")
    condition_columns: Optional[str] = Field(None, description="Condition column name")


@app.post("/generate/make-prompts", response_model=JobStatus)
async def make_prompts(request: MakePromptsRequest, background_tasks: BackgroundTasks):
    """Create prompts for generation"""
    job_id = str(uuid.uuid4())
    
    cmd = [
        "python", "-m", "_utils._generating.make_prompts",
        "--output_parquet", request.output_parquet,
        "--level", request.level
    ]
    
    if request.manual:
        cmd.append("--manual")
        if request.compositions:
            cmd.extend(["--compositions", request.compositions])
        if request.condition_lists:
            for cond in request.condition_lists:
                cmd.extend(["--condition_lists", cond])
        if request.spacegroups:
            cmd.extend(["--spacegroups", request.spacegroups])
    
    if request.automatic:
        cmd.append("--automatic")
        if request.HF_dataset:
            cmd.extend(["--HF_dataset", request.HF_dataset])
        if request.split:
            cmd.extend(["--split", request.split])
        if request.condition_columns:
            cmd.extend(["--condition_columns", request.condition_columns])
    
    background_tasks.add_task(run_command, job_id, cmd, request.output_parquet)
    
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        command=" ".join(cmd),
        started_at=datetime.now().isoformat()
    )
    
    return jobs[job_id]


class GenerateCIFsRequest(BaseModel):
    config_file: str = Field(..., description="Path to generation config JSONC file")


@app.post("/generate/cifs", response_model=JobStatus)
async def generate_cifs(request: GenerateCIFsRequest, background_tasks: BackgroundTasks):
    """Generate CIF structures from prompts"""
    job_id = str(uuid.uuid4())
    
    cmd = [
        "python", "-m", "_utils._generating.generate_CIFs",
        "--config", request.config_file
    ]
    
    background_tasks.add_task(run_command, job_id, cmd, None)
    
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        command=" ".join(cmd),
        started_at=datetime.now().isoformat()
    )
    
    return jobs[job_id]


class EvaluateCIFsRequest(BaseModel):
    input_parquet: str = Field(..., description="Input parquet with generated CIFs")
    num_workers: int = Field(8, description="Number of parallel workers")
    save_valid_parquet: bool = Field(True, description="Save valid structures")


@app.post("/generate/evaluate-cifs", response_model=JobStatus)
async def evaluate_cifs(request: EvaluateCIFsRequest, background_tasks: BackgroundTasks):
    """Evaluate structural validity of generated CIFs"""
    job_id = str(uuid.uuid4())
    
    cmd = [
        "python", "-m", "_utils._generating.evaluate_CIFs",
        "--input_parquet", request.input_parquet,
        "--num_workers", str(request.num_workers)
    ]
    
    if request.save_valid_parquet:
        cmd.append("--save_valid_parquet")
    
    background_tasks.add_task(run_command, job_id, cmd, None)
    
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        command=" ".join(cmd),
        started_at=datetime.now().isoformat()
    )
    
    return jobs[job_id]


class PostprocessRequest(BaseModel):
    input_parquet: str = Field(..., description="Input parquet with generated CIFs")
    output_parquet: str = Field(..., description="Output parquet file")
    num_workers: int = Field(4, description="Number of parallel workers")


@app.post("/generate/postprocess", response_model=JobStatus)
async def postprocess(request: PostprocessRequest, background_tasks: BackgroundTasks):
    """Post-process generated CIFs to standard format"""
    job_id = str(uuid.uuid4())
    
    cmd = [
        "python", "-m", "_utils._generating.postprocess",
        "--input_parquet", request.input_parquet,
        "--output_parquet", request.output_parquet,
        "--num_workers", str(request.num_workers)
    ]
    
    background_tasks.add_task(run_command, job_id, cmd, request.output_parquet)
    
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        command=" ".join(cmd),
        started_at=datetime.now().isoformat()
    )
    
    return jobs[job_id]


# ==================== EVALUATION METRICS ====================

class VUNMetricsRequest(BaseModel):
    gen_data: str = Field(..., description="Path to generated structures parquet")
    huggingface_dataset: str = Field(..., description="HuggingFace dataset for novelty comparison")
    output_csv: str = Field(..., description="Output CSV file")
    num_workers: int = Field(8, description="Number of parallel workers")


@app.post("/metrics/vun", response_model=JobStatus)
async def vun_metrics(request: VUNMetricsRequest, background_tasks: BackgroundTasks):
    """Calculate Validity, Uniqueness, and Novelty metrics"""
    job_id = str(uuid.uuid4())
    
    cmd = [
        "python", "_utils/_metrics/VUN_metrics.py",
        "--gen_data", request.gen_data,
        "--huggingface_dataset", request.huggingface_dataset,
        "--output_csv", request.output_csv,
        "--num_workers", str(request.num_workers)
    ]
    
    background_tasks.add_task(run_command, job_id, cmd, request.output_csv)
    
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        command=" ".join(cmd),
        started_at=datetime.now().isoformat()
    )
    
    return jobs[job_id]


class EHullMetricsRequest(BaseModel):
    post_parquet: str = Field(..., description="Path to postprocessed structures")
    output_parquet: str = Field(..., description="Output parquet with stability metrics")
    num_workers: int = Field(4, description="Number of parallel workers")


@app.post("/metrics/ehull", response_model=JobStatus)
async def ehull_metrics(request: EHullMetricsRequest, background_tasks: BackgroundTasks):
    """Calculate Energy Above Hull (thermodynamic stability)"""
    job_id = str(uuid.uuid4())
    
    cmd = [
        "python", "_utils/_metrics/mace_ehull.py",
        "--post_parquet", request.post_parquet,
        "--output_parquet", request.output_parquet,
        "--num_workers", str(request.num_workers)
    ]
    
    background_tasks.add_task(run_command, job_id, cmd, request.output_parquet)
    
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        command=" ".join(cmd),
        started_at=datetime.now().isoformat()
    )
    
    return jobs[job_id]


# ==================== JOB MANAGEMENT ====================

@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get status of a background job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/jobs", response_model=List[JobStatus])
async def list_jobs():
    """List all jobs"""
    return list(jobs.values())


# ==================== HELPER FUNCTIONS ====================

async def run_command(job_id: str, cmd: List[str], output_file: Optional[str]):
    """Execute command and update job status"""
    jobs[job_id].status = "running"
    
    try:
        # Set working directory to /app to ensure proper paths
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd="/app"
        )
        
        jobs[job_id].status = "completed"
        jobs[job_id].completed_at = datetime.now().isoformat()
        jobs[job_id].output_file = output_file
        
    except subprocess.CalledProcessError as e:
        jobs[job_id].status = "failed"
        jobs[job_id].completed_at = datetime.now().isoformat()
        jobs[job_id].error = f"Command failed: {e.stderr}\nStdout: {e.stdout}"
    except Exception as e:
        jobs[job_id].status = "failed"
        jobs[job_id].completed_at = datetime.now().isoformat()
        jobs[job_id].error = str(e)


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "CrystaLLM-2.0 API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "preprocessing": [
                "/preprocessing/deduplicate",
                "/preprocessing/clean",
                "/preprocessing/save-dataset"
            ],
            "training": ["/train"],
            "generation": [
                "/generate/direct",
                "/generate/make-prompts",
                "/generate/cifs",
                "/generate/evaluate-cifs",
                "/generate/postprocess"
            ],
            "metrics": [
                "/metrics/vun",
                "/metrics/ehull"
            ],
            "jobs": [
                "/jobs",
                "/jobs/{job_id}"
            ]
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
