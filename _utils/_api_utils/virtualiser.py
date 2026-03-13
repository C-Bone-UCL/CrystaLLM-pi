"""Virtualiser API endpoint: convert ordered CIF structures to disordered virtual crystals."""

import uuid
import yaml
from pathlib import Path
from typing import Callable, List, Optional, Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field


class VirtualiseRequest(BaseModel):
    input_cif: str = Field(..., description="Path to input ordered CIF file.")
    output_cif: str = Field(..., description="Path for the output virtual crystal CIF.")
    config_file: Optional[str] = Field(
        None,
        description=(
            "Path to a YAML config file specifying virtual_pairs, symprec, and angle_tolerance. "
            "If omitted, provide virtual_pairs directly."
        ),
    )
    virtual_pairs: Optional[List[List[str]]] = Field(
        None,
        description=(
            'Element pairs to virtualise, e.g. [["Mg", "Zn"]]. '
            "Required when config_file is not provided."
        ),
    )
    symprec: float = Field(0.003, description="Symmetry precision in Å (used when config_file is not provided).")
    angle_tolerance: float = Field(0.5, description="Angle tolerance in degrees (used when config_file is not provided).")


def register_virtualiser_routes(
    app: FastAPI,
    create_pending_job: Callable[[str, List[str]], Any],
    run_command: Callable[[str, List[str], Optional[str]], None],
) -> None:
    """Register virtualiser endpoints on the API app."""

    @app.post("/virtualise")
    async def virtualise(request: VirtualiseRequest, background_tasks: BackgroundTasks):
        """Convert an ordered CIF to a disordered virtual crystal with promoted symmetry."""
        if request.config_file is None and not request.virtual_pairs:
            raise HTTPException(
                status_code=422,
                detail="Provide either config_file or virtual_pairs.",
            )

        job_id = str(uuid.uuid4())

        # If no config file supplied, write a temporary one from inline params.
        if request.config_file is None:
            tmp_config = Path(f"/tmp/{job_id}_virtualiser_config.yaml")
            pairs = [list(p) for p in request.virtual_pairs]
            cfg = {
                "symprec": request.symprec,
                "angle_tolerance": request.angle_tolerance,
                "virtual_pairs": pairs,
            }
            tmp_config.write_text(yaml.dump(cfg), encoding="utf-8")
            config_path = str(tmp_config)
        else:
            config_path = request.config_file

        cmd = [
            "python", "-m", "_utils._virtualiser.crystal_virtualiser",
            "--in", request.input_cif,
            "--config", config_path,
            "--out", request.output_cif,
        ]

        background_tasks.add_task(run_command, job_id, cmd, request.output_cif)
        return create_pending_job(job_id, cmd)
