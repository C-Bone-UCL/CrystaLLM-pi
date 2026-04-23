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

    def test_vun_metrics_with_optional_flags(self):
        """Test VUN metrics forwards maintained optional CLI flags."""
        if self._should_skip_integration():
            return

        gen_data = self._out("postprocessed.parquet") if self.is_integration else self.test_data['test_file']
        output_csv = self._out("vun_metrics_extended.csv") if self.is_integration else os.path.join(self.temp_dir, "vun_extended.csv")
        output_parquet = self._out("vun_metrics_extended.parquet") if self.is_integration else os.path.join(self.temp_dir, "vun_extended.parquet")
        response = self.client.post("/metrics/vun", json={
            "gen_data": gen_data,
            "huggingface_dataset": "c-bone/SLME_1K_dtest",
            "output_csv": output_csv,
            "output_parquet": output_parquet,
            "sort_metrics_by": "both",
            "num_workers": 4,
            "load_processed_data": "/tmp/processed_reference.parquet",
            "check_comp_novelty": True,
            "skip_validity": True,
            "skip_uniqueness": True,
            "bond_length_acceptability_cutoff": 0.95,
            "allow_stated_p1_mismatch": True
        })
        data = self._wait_and_assert(response, job_name="vun_metrics_optional", timeout=600)
        assert "--output_parquet" in data["command"]
        assert output_parquet in data["command"]
        assert "--sort_metrics_by both" in data["command"]
        assert "--load_processed_data /tmp/processed_reference.parquet" in data["command"]
        assert "--check_comp_novelty" in data["command"]
        assert "--skip_validity" in data["command"]
        assert "--skip_uniqueness" in data["command"]
        assert "--bond_length_acceptability_cutoff 0.95" in data["command"]
        assert "--allow_stated_p1_mismatch" in data["command"]

    def test_vun_metrics_rejects_invalid_sort_metrics_by(self):
        """Test VUN metrics rejects unsupported grouping modes."""
        response = self.client.post("/metrics/vun", json={
            "gen_data": self.test_data['test_file'],
            "huggingface_dataset": "c-bone/SLME_1K_dtest",
            "output_csv": os.path.join(self.temp_dir, "vun_bad.csv"),
            "sort_metrics_by": "invalid"
        })
        assert response.status_code == 422
        
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

    def test_ehull_metrics_with_optional_flags(self):
        """Test E-hull metrics forwards maintained optional CLI flags."""
        if self._should_skip_integration():
            return

        post_parquet = self._out("postprocessed.parquet") if self.is_integration else self.test_data['test_file']
        output_parquet = self._out("ehull_extended.parquet") if self.is_integration else os.path.join(self.temp_dir, "ehull_extended.parquet")
        response = self.client.post("/metrics/ehull", json={
            "post_parquet": post_parquet,
            "output_parquet": output_parquet,
            "mp_data": "/tmp/mp_data.json.gz",
            "cif_column": "CIF",
            "num_workers": 2,
            "batch_size": 16
        })
        data = self._wait_and_assert(response, job_name="ehull_metrics_optional", timeout=600)
        assert "--mp_data /tmp/mp_data.json.gz" in data["command"]
        assert "--cif_column CIF" in data["command"]
        assert "--batch_size 16" in data["command"]

    def test_xrd_metrics_endpoint(self):
        """Smoke test XRD metrics endpoint command construction."""
        if self.is_integration:
            return

        output_parquet = os.path.join(self.temp_dir, "xrd_metrics.parquet")
        response = self.client.post("/metrics/xrd", json={
            "input_parquet": self.test_data['test_file'],
            "output_parquet": output_parquet,
            "num_gens": 1,
            "num_workers": 2,
            "ref_parquet": "/tmp/reference_structures.parquet",
            "validity_check": "crystallm",
            "sort_gens": "random"
        })
        assert response.status_code == 200
        data = response.json()
        assert "XRD_metrics" in data["command"]
        assert "--input_parquet" in data["command"]
        assert "--output_parquet" in data["command"]
        assert "--num_gens 1" in data["command"]
        assert "--ref_parquet /tmp/reference_structures.parquet" in data["command"]
        assert "--validity_check crystallm" in data["command"]
        assert "--sort_gens random" in data["command"]

    def test_property_metrics_endpoint(self):
        """Smoke test property metrics endpoint command construction."""
        if self.is_integration:
            return

        parquet_out = os.path.join(self.temp_dir, "property_metrics.parquet")
        metrics_out = os.path.join(self.temp_dir, "property_metrics.csv")
        response = self.client.post("/metrics/property", json={
            "post_parquet": self.test_data['test_file'],
            "parquet_out": parquet_out,
            "metrics_out": metrics_out,
            "sort_metrics_by": "both",
            "num_workers": 2,
            "property_targets": "[\"Bandgap (eV)\"]",
            "property1_normaliser": "power_log",
            "max_property1": 10.0,
            "min_property1": 0.0
        })
        assert response.status_code == 200
        data = response.json()
        assert "property_metrics" in data["command"]
        assert "--post_parquet" in data["command"]
        assert "--parquet_out" in data["command"]
        assert "--metrics_out" in data["command"]
        assert "--sort_metrics_by both" in data["command"]
        assert "--property_targets [\"Bandgap (eV)\"]" in data["command"]
        assert "--property1_normaliser power_log" in data["command"]
        assert "--max_property1 10.0" in data["command"]
