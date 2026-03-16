"""API endpoint tests: virtualiser."""

import os

from tests.api.endpoints_sections._base import IntegrationMixin


class VirtualiserEndpointTests(IntegrationMixin):
    """Test the /virtualise endpoint."""

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

    def test_virtualise_with_inline_pairs(self):
        """Endpoint accepts inline virtual_pairs and returns a pending job."""
        output_cif = os.path.join(self.temp_dir, "virtual_out.cif")
        response = self.client.post("/virtualise", json={
            "input_cif": self.test_data.get("test_cif_path", "/app/data/test.cif"),
            "output_cif": output_cif,
            "virtual_pairs": [["Mg", "Zn"]],
            "symprec": 0.003,
            "angle_tolerance": 0.5,
        })
        data = self._wait_and_assert(response, job_name="virtualise_inline_pairs")
        assert "_virtualiser.crystal_virtualiser" in data["command"]
        assert "--in" in data["command"]
        assert "--out" in data["command"]

    def test_virtualise_with_config_file(self):
        """Endpoint accepts a config_file path and constructs the correct command."""
        output_cif = os.path.join(self.temp_dir, "virtual_out_cfg.cif")
        response = self.client.post("/virtualise", json={
            "input_cif": "/app/data/test.cif",
            "output_cif": output_cif,
            "config_file": "/app/data/config.yaml",
        })
        data = self._wait_and_assert(response, job_name="virtualise_config_file")
        assert "_virtualiser.crystal_virtualiser" in data["command"]
        assert "--config" in data["command"]

    def test_virtualise_missing_pairs_and_config(self):
        """Endpoint returns 422 when neither config_file nor virtual_pairs are provided."""
        response = self.client.post("/virtualise", json={
            "input_cif": "/app/data/test.cif",
            "output_cif": "/app/outputs/virtual.cif",
        })
        assert response.status_code == 422

    def test_virtualise_missing_required_fields(self):
        """Endpoint returns 422 when input_cif or output_cif are missing."""
        response = self.client.post("/virtualise", json={
            "virtual_pairs": [["Mg", "Zn"]],
        })
        assert response.status_code == 422
