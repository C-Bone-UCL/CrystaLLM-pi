"""API endpoint tests: metrics."""

import os

from tests.api.endpoints_sections._base import IntegrationMixin

class MetricsEndpointTests(IntegrationMixin):
    """Test metrics endpoints."""
    
    def __init__(
        self,
        client,
        temp_dir: str,
        test_data: dict,
        mode: str = "smoke",
        docker_mode: bool = False,
        verbose: bool = False,
        integration_tag: str | None = None,
    ):
        self._init_integration(
            client=client,
            temp_dir=temp_dir,
            test_data=test_data,
            mode=mode,
            docker_mode=docker_mode,
            verbose=verbose,
            integration_tag=integration_tag,
        )
        
    def test_vun_metrics(self):
        """Test VUN metrics endpoint."""
        if self._should_skip_integration():
            return

        gen_data = self._out("postprocessed.parquet") if self.is_integration else self.test_data['test_file']
        if self.is_integration:
            self._assert_output_exists(gen_data, label="VUN input parquet")
        output_csv = self._out("vun_metrics.csv") if self.is_integration else os.path.join(self.temp_dir, "vun.csv")
        response = self.client.post("/metrics/vun", json={
            "gen_data": gen_data,
            "huggingface_dataset": "c-bone/SLME_1K_dtest",
            "output_csv": output_csv,
            "num_workers": 4
        })
        data = self._wait_and_assert(response, job_name="vun_metrics", timeout=600)
        assert "VUN_metrics" in data["command"]
        assert "--huggingface_dataset" in data["command"]

        if self.is_integration:
            self._assert_output_exists(output_csv, label="VUN metrics CSV")

        if self.is_integration and self.verbose:
            self._show_vun_metrics(output_csv)
        
    def test_ehull_metrics(self):
        """Test energy above hull metrics endpoint."""
        if self._should_skip_integration():
            return

        post_parquet = self._out("postprocessed.parquet") if self.is_integration else self.test_data['test_file']
        if self.is_integration:
            self._assert_output_exists(post_parquet, label="E-hull input parquet")
        output_parquet = self._out("ehull_results.parquet") if self.is_integration else os.path.join(self.temp_dir, "ehull.parquet")
        response = self.client.post("/metrics/ehull", json={
            "post_parquet": post_parquet,
            "output_parquet": output_parquet,
            "num_workers": 2
        })
        data = self._wait_and_assert(response, job_name="ehull_metrics", timeout=600)
        assert "mace_ehull" in data["command"]

        if self.is_integration:
            self._assert_output_exists(output_parquet, label="E-hull output parquet")

        if self.is_integration and self.verbose:
            self._show_ehull_stats(output_parquet)
