"""API endpoint tests: root."""

class RootEndpointTests:
    """Test the root endpoint."""
    
    def __init__(self, client, temp_dir: str):
        self.client = client
        self.temp_dir = temp_dir
        
    def test_root_returns_api_info(self):
        """Test that root endpoint returns API info."""
        response = self.client.get("/")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert "name" in data, "Response should contain 'name'"
        assert "version" in data, "Response should contain 'version'"
        assert "endpoints" in data, "Response should contain 'endpoints'"
        
    def test_root_lists_all_endpoint_categories(self):
        """Test that root lists all endpoint categories."""
        response = self.client.get("/")
        data = response.json()
        
        expected_categories = ["health", "preprocessing", "training", "generation", "metrics", "jobs"]
        for cat in expected_categories:
            assert cat in data["endpoints"], f"Missing category: {cat}"

    def test_health_endpoint(self):
        """Test that health endpoint returns a healthy status payload."""
        response = self.client.get("/healthz")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert data.get("status") == "ok", "Health status should be 'ok'"
        assert "service" in data, "Response should contain 'service'"
