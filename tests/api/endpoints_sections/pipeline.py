"""API endpoint tests: full integration pipeline."""

from tests.api.endpoints_sections._base import IntegrationMixin


class PipelineIntegrationTests(IntegrationMixin):
    """Integration-only full pipeline test."""

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

    def test_full_generation_pipeline(self):
        """Run generation -> evaluate -> postprocess -> metrics in one flow."""
        if not self.is_integration:
            return
        if self._should_skip_integration():
            return

        print("\n      Full pipeline test")

        generated_path = self._out("pipeline_generated.parquet")
        valid_path = self._out("pipeline_valid.parquet")
        postprocessed_path = self._out("pipeline_postprocessed.parquet")
        vun_path = self._out("pipeline_vun.csv")
        ehull_path = self._out("pipeline_ehull.parquet")

        print("      Step 1/5 Generating structures")
        # Add target_valid_cifs to guarantee output
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_bandgap",
            "output_parquet": generated_path,
            "reduced_formula_list": "Si,GaAs",
            "z_list": "2,1",
            "condition_lists": ["1.1,0.0", "1.4,0.0"],
            "level": "level_2",
            "num_return_sequences": 5,
            "max_return_attempts": 5,
            "target_valid_cifs": 1
        })

        if response.status_code == 422:
            print("\n422 SCHEMA ERROR DETAILS:")
            print(response.json())
        self._wait_and_assert(response, job_name="generate", timeout=900)
        self._assert_output_exists(generated_path, label="pipeline generated parquet")

        if self.verbose:
            self._show_generated_cifs(generated_path, max_cifs=3)

        print("      Step 2/5 Evaluating structures")
        response = self.client.post("/generate/evaluate-cifs", json={
            "input_parquet": generated_path,
            "num_workers": 4,
            "save_valid_parquet": valid_path
        })
        self._wait_and_assert(response, job_name="evaluate")

        print("      Step 3/5 Postprocessing")
        response = self.client.post("/generate/postprocess", json={
            "input_parquet": generated_path,
            "output_parquet": postprocessed_path,
            "num_workers": 4
        })
        self._wait_and_assert(response, job_name="postprocess")
        self._assert_output_exists(postprocessed_path, label="pipeline postprocessed parquet")

        print("      Step 4/5 Computing VUN metrics")
        response = self.client.post("/metrics/vun", json={
            "gen_data": postprocessed_path,
            "huggingface_dataset": "c-bone/SLME_1K_dtest",
            "output_csv": vun_path,
            "num_workers": 4
        })
        self._wait_and_assert(response, job_name="vun", timeout=600)
        self._assert_output_exists(vun_path, label="pipeline VUN CSV")

        if self.verbose:
            self._show_vun_metrics(vun_path)
            self._show_validity_stats(postprocessed_path)

        print("      Step 5/5 Computing E-hull metrics")
        response = self.client.post("/metrics/ehull", json={
            "post_parquet": postprocessed_path,
            "output_parquet": ehull_path,
            "num_workers": 2
        })
        self._wait_and_assert(response, job_name="ehull", timeout=600)
        self._assert_output_exists(ehull_path, label="pipeline E-hull parquet")

        if self.verbose:
            self._show_ehull_stats(ehull_path)

        print("      Pipeline complete")