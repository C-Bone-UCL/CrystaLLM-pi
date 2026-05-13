"""Generation API endpoints."""

import uuid
from typing import Callable, Optional, List, Literal, Any, Union

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field


def _split_csv_values(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


class DirectGenerationRequest(BaseModel):
    """Request model for direct HuggingFace model generation."""

    hf_model_path: str = Field(..., description="HuggingFace model path (e.g., c-bone/CrystaLLM-pi_bandgap)")
    output_parquet: Optional[str] = Field(None, description="Output parquet file path")
    output_cif_dir: Optional[str] = Field(None, description="Output directory for generated CIF files")
    input_parquet: Optional[str] = Field(None, description="Input prompts parquet (alternative to manual)")
    reduced_formula_list: Optional[str] = Field(None, description="Comma-separated reduced formulas")
    search_zs: bool = Field(False, description="Search through Z=1 to Z=4 to find valid structures")
    z_list: Optional[str] = Field(None, description="Comma-separated explicit Z integers mapping 1:1 to formulas")
    condition_lists: Optional[List[str]] = Field(None, description="Condition vectors")
    xrd_files: Optional[List[str]] = Field(None, description="Files with XRD peaks (.csv, .xy, .txt, .dat) for XRD models")
    xrd_wavelength: float = Field(1.54056, description="Wavelength of the provided XRD data (default CuKa)")
    level: Literal["level_1", "level_2", "level_3", "level_4"] = Field("level_2", description="Prompt detail level")
    spacegroups: Optional[str] = Field(None, description="Comma-separated spacegroups (level_4 only)")
    temperature: float = Field(1.0, description="Sampling temperature")
    num_return_sequences: int = Field(1, description="Sequences per sample")
    max_return_attempts: int = Field(1, description="Generation attempts per sample")
    max_samples: Optional[int] = Field(None, description="Max samples to process")
    scoring_mode: str = Field("None", description="Scoring mode for filtering (case-insensitive: LOGP or None)")
    target_valid_cifs: int = Field(1, description="Target valid CIFs per prompt (LOGP requires sensibility + formula-structure consistency)")
    multi_gpu: Literal["auto", "true", "false"] = Field("auto", description="Generation launcher mode")
    nproc_per_node: Optional[int] = Field(None, description="Max GPU workers when multi_gpu is enabled (will be passed as --num_workers_gpu)")
    num_workers: int = Field(4, description="Post-processing workers")
    skip_postprocess: bool = Field(False, description="Skip CIF validation/postprocessing")


class MakePromptsRequest(BaseModel):
    output_parquet: str = Field(..., description="Output parquet file")
    manual: Optional[bool] = Field(None, description="Manual mode")
    compositions: Optional[str] = Field(None, description="Comma-separated compositions")
    condition_lists: Optional[List[str]] = Field(None, description="Condition value lists")
    level: Literal["level_1", "level_2", "level_3", "level_4"] = Field("level_2", description="Prompt level")
    raw: bool = Field(False, description="Use raw conditioning format")
    spacegroups: Optional[str] = Field(None, description="Space groups (for level_4)")
    automatic: bool = Field(False, description="Automatic mode (extract from dataset)")
    HF_dataset: Optional[str] = Field(None, description="HuggingFace dataset name")
    split: Optional[str] = Field(None, description="Dataset split")
    cif_column: str = Field("CIF", description="Column containing CIF strings for automatic mode")
    condition_columns: Optional[Union[str, List[str]]] = Field(
        None,
        description="Condition column name(s), either a single string or a list",
    )
    input_parquet: Optional[str] = Field(None, description="Input parquet (alternative to HF dataset)")
    remove_ref_columns: bool = Field(False, description="Keep only prompt and condition columns in automatic output")
    mode: Literal["cartesian", "paired", "broadcast"] = Field("cartesian", description="Composition-condition pairing mode for manual prompts")


class GenerateCIFsRequest(BaseModel):
    config_file: str = Field(..., description="Path to generation config JSONC file")


class EvaluateCIFsRequest(BaseModel):
    input_parquet: str = Field(..., description="Input parquet with generated CIFs")
    metrics_out: Optional[str] = Field(None, description="Path to save evaluation metrics parquet")
    num_workers: int = Field(8, description="Number of parallel workers")
    save_valid_parquet: Optional[str] = Field(None, description="Path to save valid structures parquet")
    debug: bool = Field(False, description="Enable debug output during evaluation")


class PostprocessRequest(BaseModel):
    input_parquet: str = Field(..., description="Input parquet with generated CIFs")
    output_parquet: str = Field(..., description="Output parquet file")
    num_workers: int = Field(4, description="Number of parallel workers")
    column_name: str = Field("Generated CIF", description="Column containing CIF strings to postprocess")


def register_generation_routes(
    app: FastAPI,
    create_pending_job: Callable[[str, List[str]], Any],
    run_command: Callable[[str, List[str], Optional[str]], None],
) -> None:
    """Register generation endpoints on the API app."""

    @app.post("/generate/direct")
    async def generate_direct(request: DirectGenerationRequest, background_tasks: BackgroundTasks):
        if bool(request.output_parquet) == bool(request.output_cif_dir):
            raise HTTPException(
                status_code=422,
                detail="Provide exactly one output target: output_parquet or output_cif_dir.",
            )

        normalized_scoring_mode = str(request.scoring_mode).lower()
        if normalized_scoring_mode not in {"logp", "none"}:
            raise HTTPException(status_code=422, detail="scoring_mode must be LOGP or None.")
        
        # Auto-inject a dummy formula for level_1 if no inputs are provided
        if not request.input_parquet and not request.reduced_formula_list:
            if request.level == "level_1":
                request.reduced_formula_list = "X"
                request.z_list = "1"
            else:
                raise HTTPException(status_code=422, detail="Must provide input_parquet or reduced_formula_list for level_2+ generation.")

        if request.reduced_formula_list and request.input_parquet:
            raise HTTPException(status_code=422, detail="Use either input_parquet or reduced_formula_list mode, not both")

        if request.search_zs and request.z_list:
            raise HTTPException(status_code=422, detail="search_zs and z_list are mutually exclusive.")

        if request.reduced_formula_list:
            formula_values = _split_csv_values(request.reduced_formula_list)
            formula_count = len(formula_values)

            if request.xrd_files and len(request.xrd_files) != formula_count:
                raise HTTPException(status_code=422, detail="XRD files must map 1:1 to reduced formulas.")

            if request.z_list and len(_split_csv_values(request.z_list)) != formula_count:
                raise HTTPException(status_code=422, detail="z_list must map 1:1 to reduced formulas.")

            if request.spacegroups:
                spacegroup_values = _split_csv_values(request.spacegroups)
                if len(spacegroup_values) not in {1, formula_count}:
                    raise HTTPException(status_code=422, detail="spacegroups must provide either one value or one per reduced formula.")

            if request.condition_lists and "xrd" not in request.hf_model_path.lower():
                if len(request.condition_lists) not in {1, formula_count}:
                    raise HTTPException(status_code=422, detail="condition vector mappings must provide either one vector or one per reduced formula.")

        if normalized_scoring_mode == "logp" and request.target_valid_cifs == 0:
            raise HTTPException(
                status_code=422,
                detail="scoring_mode='LOGP' requires target_valid_cifs > 0.",
            )

        job_id = str(uuid.uuid4())
        cmd = [
            "python", "-m", "_load_and_generate",
            "--hf_model_path", request.hf_model_path,
        ]

        if request.output_parquet:
            cmd.extend(["--output_parquet", request.output_parquet])
        else:
            cmd.extend(["--output_cif_dir", request.output_cif_dir])

        if request.input_parquet:
            cmd.extend(["--input_parquet", request.input_parquet])
        else:
            cmd.extend(["--reduced_formula_list", request.reduced_formula_list])
            if request.search_zs:
                cmd.append("--search_zs")
            if request.z_list:
                cmd.extend(["--z_list", request.z_list])
            if request.xrd_files:
                cmd.append("--xrd_files")
                cmd.extend(request.xrd_files)
                cmd.extend(["--xrd_wavelength", str(request.xrd_wavelength)])
            if request.condition_lists:
                cmd.append("--condition_lists")
                cmd.extend(request.condition_lists)
            if request.spacegroups:
                cmd.extend(["--spacegroups", request.spacegroups])
            cmd.extend(["--level", request.level])

        cmd.extend([
            "--temperature", str(request.temperature),
            "--num_return_sequences", str(request.num_return_sequences),
            "--max_return_attempts", str(request.max_return_attempts),
            "--scoring_mode", request.scoring_mode,
            "--multi_gpu", request.multi_gpu,
            "--target_valid_cifs", str(request.target_valid_cifs),
            "--num_workers", str(request.num_workers),
        ])

        if request.max_samples is not None:
            cmd.extend(["--max_samples", str(request.max_samples)])
        if request.nproc_per_node is not None:
            cmd.extend(["--num_workers_gpu", str(request.nproc_per_node)])

        if request.skip_postprocess:
            cmd.append("--skip_postprocess")

        output_target = request.output_parquet or request.output_cif_dir
        background_tasks.add_task(run_command, job_id, cmd, output_target)
        return create_pending_job(job_id, cmd)

    @app.post("/generate/make-prompts")
    async def make_prompts(request: MakePromptsRequest, background_tasks: BackgroundTasks):
        is_manual = request.manual is True
        is_automatic = request.automatic is True
        if is_manual == is_automatic:
            raise HTTPException(
                status_code=422,
                detail="Choose exactly one mode for /generate/make-prompts: automatic or manual.",
            )

        if is_automatic and not request.input_parquet and not request.HF_dataset:
            raise HTTPException(
                status_code=422,
                detail="Automatic make-prompts requires input_parquet or HF_dataset.",
            )

        if is_automatic and request.HF_dataset and not request.split:
            raise HTTPException(
                status_code=422,
                detail="split is required when using HF_dataset in automatic make-prompts mode.",
            )

        job_id = str(uuid.uuid4())
        cmd = [
            "python", "-m", "_utils._generating.make_prompts",
            "--output_parquet", request.output_parquet,
            "--level", request.level,
        ]

        if request.raw:
            cmd.append("--raw")

        if is_manual:
            cmd.append("--manual")
            if request.compositions:
                cmd.extend(["--compositions", request.compositions])
            if request.condition_lists:
                for cond in request.condition_lists:
                    cmd.extend(["--condition_lists", cond])
            if request.spacegroups:
                cmd.extend(["--spacegroups", request.spacegroups])
            cmd.extend(["--mode", request.mode])

        if is_automatic:
            cmd.append("--automatic")
            if request.input_parquet:
                cmd.extend(["--input_parquet", request.input_parquet])
            if request.HF_dataset:
                cmd.extend(["--HF_dataset", request.HF_dataset])
            if request.split:
                cmd.extend(["--split", request.split])
            cmd.extend(["--cif_column", request.cif_column])
            if request.condition_columns:
                if isinstance(request.condition_columns, list):
                    condition_columns = [value for value in request.condition_columns if value]
                else:
                    condition_columns = [request.condition_columns] if request.condition_columns else []

                if condition_columns:
                    cmd.append("--condition_columns")
                    cmd.extend(condition_columns)
            if request.remove_ref_columns:
                cmd.append("--remove_ref_columns")

        background_tasks.add_task(run_command, job_id, cmd, request.output_parquet)
        return create_pending_job(job_id, cmd)

    @app.post("/generate/cifs")
    async def generate_cifs(request: GenerateCIFsRequest, background_tasks: BackgroundTasks):
        job_id = str(uuid.uuid4())
        cmd = [
            "python", "-m", "_utils._generating.generate_CIFs",
            "--config", request.config_file,
        ]

        background_tasks.add_task(run_command, job_id, cmd, None)
        return create_pending_job(job_id, cmd)

    @app.post("/generate/evaluate-cifs")
    async def evaluate_cifs(request: EvaluateCIFsRequest, background_tasks: BackgroundTasks):
        job_id = str(uuid.uuid4())
        cmd = [
            "python", "-m", "_utils._generating.evaluate_CIFs",
            "--input_parquet", request.input_parquet,
            "--num_workers", str(request.num_workers),
        ]

        if request.metrics_out:
            cmd.extend(["--metrics_out", request.metrics_out])
        if request.save_valid_parquet:
            cmd.extend(["--save_valid_parquet", request.save_valid_parquet])
        if request.debug:
            cmd.append("--debug")

        background_tasks.add_task(run_command, job_id, cmd, None)
        return create_pending_job(job_id, cmd)

    @app.post("/generate/postprocess")
    async def postprocess(request: PostprocessRequest, background_tasks: BackgroundTasks):
        job_id = str(uuid.uuid4())
        cmd = [
            "python", "-m", "_utils._generating.postprocess",
            "--input_parquet", request.input_parquet,
            "--output_parquet", request.output_parquet,
            "--num_workers", str(request.num_workers),
            "--column_name", request.column_name,
        ]

        background_tasks.add_task(run_command, job_id, cmd, request.output_parquet)
        return create_pending_job(job_id, cmd)
