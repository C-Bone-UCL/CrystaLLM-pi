"""API endpoint tests: api gaps."""

class APIGapTests:
    """Test for known gaps and issues in the API."""
    
    def __init__(self, client, temp_dir: str):
        self.client = client
        self.temp_dir = temp_dir
        
    def test_xrd_preprocessing_endpoint_now_supported(self):
        """XRD preprocessing endpoint is now exposed in the API."""
        response = self.client.get("/")
        endpoints = response.json()["endpoints"]["preprocessing"]
        assert "/preprocessing/process-exp-xrd" in endpoints, "XRD preprocessing endpoint should be listed"
        
    def test_xrd_metrics_endpoint_now_supported(self):
        """XRD metrics are now exposed in the API."""
        response = self.client.get("/")
        endpoints = response.json()["endpoints"]["metrics"]
        assert "/metrics/xrd" in endpoints, "XRD metrics endpoint should be listed"
        
    def test_property_metrics_endpoint_now_supported(self):
        """Property metrics are now exposed in the API."""
        response = self.client.get("/")
        endpoints = response.json()["endpoints"]["metrics"]
        assert "/metrics/property" in endpoints, "Property metrics endpoint should be listed"