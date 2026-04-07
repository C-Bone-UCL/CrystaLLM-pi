"""API endpoint tests: jobs."""

class JobManagementTests:
    """Test job management endpoints."""
    
    def __init__(self, client, temp_dir: str):
        self.client = client
        self.temp_dir = temp_dir
        
    def test_list_jobs_empty(self):
        """Test listing jobs when none exist."""
        response = self.client.get("/jobs")
        assert response.status_code == 200
        # can be empty or have jobs from other tests
        
    def test_get_nonexistent_job_returns_404(self):
        """Test getting a job that doesn't exist."""
        response = self.client.get("/jobs/nonexistent-job-id-12345")
        assert response.status_code == 404, f"Expected 404, got {response.status_code}"
        
    def test_job_creation_returns_pending_status(self):
        """Test that creating a job returns pending status."""
        # use a simple endpoint to create a job
        response = self.client.post("/preprocessing/deduplicate", json={
            "input_parquet": "/tmp/fake_input.parquet",
            "output_parquet": "/tmp/fake_output.parquet"
        })
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "pending", f"Expected 'pending', got {data['status']}"
        assert "job_id" in data, "Response should have job_id"
        assert "command" in data, "Response should have command"
        
    def test_job_can_be_retrieved_after_creation(self):
        """Test that a created job can be retrieved."""
        # create job
        response = self.client.post("/preprocessing/deduplicate", json={
            "input_parquet": "/tmp/test_input.parquet",
            "output_parquet": "/tmp/test_output.parquet"
        })
        job_id = response.json()["job_id"]
        
        # retrieve it
        response = self.client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        assert response.json()["job_id"] == job_id
