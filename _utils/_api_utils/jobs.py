"""Job management API endpoints."""

from typing import Callable, List, Any

from fastapi import FastAPI, HTTPException


def register_jobs_routes(
    app: FastAPI,
    load_job: Callable[[str], Any],
    list_jobs: Callable[[], List[Any]],
    cancel_job: Callable[[str], Any],
) -> None:
    """Register job status/list endpoints on the API app."""

    @app.get("/jobs/{job_id}")
    async def get_job_status(job_id: str):
        job = load_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.get("/jobs")
    async def list_jobs_endpoint():
        return list_jobs()

    @app.post("/jobs/{job_id}/cancel")
    async def cancel_job_endpoint(job_id: str):
        success, message = cancel_job(job_id)
        if not success:
            if message == "Job not found":
                raise HTTPException(status_code=404, detail=message)
            raise HTTPException(status_code=409, detail=message)
        return {"job_id": job_id, "status": "cancelled", "detail": message}
