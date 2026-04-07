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
        
    def test_missing_xrd_metrics_endpoint(self):
        """XRD metrics exist in CLI but not API.
        
        Script: _utils/_metrics/XRD_metrics.py
        This should be exposed as /metrics/xrd
        """
        response = self.client.get("/")
        endpoints = response.json()["endpoints"]["metrics"]
        assert "/metrics/xrd" not in endpoints, "XRD metrics endpoint is missing (expected)"
        
    def test_missing_property_metrics_endpoint(self):
        """Property metrics exist in CLI but not API.
        
        Script: _utils/_metrics/property_metrics.py (ALIGNN predictions)
        This should be exposed as /metrics/property
        """
        response = self.client.get("/")
        endpoints = response.json()["endpoints"]["metrics"]
        # only vun and ehull exist
        assert len([e for e in endpoints if "property" in e]) == 0, "Property metrics endpoint missing (expected)"