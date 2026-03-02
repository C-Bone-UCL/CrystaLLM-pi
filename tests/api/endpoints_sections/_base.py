"""Shared integration helpers for API endpoint test sections."""

import os
import time
import uuid

import pandas as pd


class IntegrationMixin:
    """Common helpers for mode-aware smoke and integration endpoint tests."""

    def _init_integration(
        self,
        *,
        client,
        temp_dir: str,
        test_data: dict,
        mode: str = "smoke",
        docker_mode: bool = False,
        verbose: bool = False,
        integration_tag: str | None = None,
    ):
        self.client = client
        self.temp_dir = temp_dir
        self.test_data = test_data
        self.mode = mode
        self.docker_mode = docker_mode
        self.verbose = verbose
        self.is_integration = mode == "integration"

        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        self.local_data_dir = os.path.join(self.project_root, "data")
        self.local_output_dir = os.path.join(self.project_root, "outputs")

        tag = integration_tag or f"api_itest_{uuid.uuid4().hex[:8]}"
        self.run_id = tag
        self.test_input_name = f"test_input_{self.run_id}.parquet"

        if self.docker_mode:
            self.data_dir = "/app/data"
            self.output_dir = f"/app/outputs/{self.run_id}"
            self.test_input_parquet = f"{self.data_dir}/{self.test_input_name}"
        else:
            self.data_dir = temp_dir
            self.output_dir = temp_dir
            self.test_input_parquet = test_data["test_file"]

        if self.is_integration and self.docker_mode:
            os.makedirs(os.path.join(self.local_output_dir, self.run_id), exist_ok=True)
            self._setup_docker_test_data()
            self._print_runtime_preflight()

    def _should_skip_integration(self) -> bool:
        if self.is_integration and not self.docker_mode:
            print("    Skipping - requires Docker environment")
            return True
        return False

    def _out(self, filename: str) -> str:
        return os.path.join(self.output_dir, filename)

    def _input_parquet(self, smoke_input: str) -> str:
        if self.is_integration:
            return self.test_input_parquet
        return smoke_input

    def _wait_and_assert(self, response, *, job_name: str, timeout: int = 300):
        assert response.status_code == 200, f"Request failed: {response.status_code}"
        if not self.is_integration:
            return response.json()
        job = self.wait_for_job(response.json()["job_id"], timeout=timeout, job_name=job_name)
        assert job["status"] == "completed", f"Job failed: {job.get('error')}"
        return job

    def _setup_docker_test_data(self):
        """Create test input parquet in host data/ directory mounted into Docker."""
        os.makedirs(self.local_data_dir, exist_ok=True)

        test_file = os.path.join(self.local_data_dir, self.test_input_name)
        if os.path.exists(test_file):
            return

        test_df = self.test_data["test_df"].copy()
        test_df.to_parquet(test_file, index=False)

    def _print_runtime_preflight(self):
        """Show runtime settings that often explain integration permission failures."""
        print("\n      Runtime preflight (Docker mode)")
        print(f"      data_dir={self.data_dir}")
        print(f"      output_dir={self.output_dir}")

        response = self.client.get("/")
        if response.status_code == 200:
            print("      API root reachable: yes")
        else:
            print(f"      API root reachable: no ({response.status_code})")

    def cleanup_artifacts(self):
        """Remove per-run host-side artifacts created for Docker integration tests."""
        if not self.docker_mode:
            return

        local_test_file = os.path.join(self.local_data_dir, self.test_input_name)
        if os.path.exists(local_test_file):
            try:
                os.remove(local_test_file)
                print(f"      Cleaned test input: {local_test_file}")
            except Exception as exc:
                print(f"      Could not remove test input {local_test_file}: {exc}")

    def _get_local_path(self, docker_path: str) -> str:
        """Convert Docker path to local path for reading outputs."""
        if docker_path.startswith("/app/outputs"):
            return docker_path.replace("/app/outputs", self.local_output_dir)
        return docker_path

    def _assert_output_exists(self, docker_path: str, label: str = "artifact"):
        """Fail fast with a clear message when an expected output was not created."""
        local_path = self._get_local_path(docker_path)
        assert os.path.exists(local_path), f"Expected {label} not found: {local_path}"

    def _show_generated_cifs(self, parquet_path: str, max_cifs: int = 2):
        """Display sample generated CIFs from parquet file."""
        local_path = self._get_local_path(parquet_path)
        if not os.path.exists(local_path):
            print(f"      [verbose] File not found: {local_path}")
            return

        df = pd.read_parquet(local_path)
        cif_col = "Generated CIF" if "Generated CIF" in df.columns else "CIF"

        print(f"\n      {'='*60}")
        print(f"      GENERATED STRUCTURES ({len(df)} total)")
        print(f"      {'='*60}")

        if "Reduced Formula" in df.columns:
            print(f"      Compositions: {df['Reduced Formula'].value_counts().to_dict()}")

        for i, row in df.head(max_cifs).iterrows():
            cif = row.get(cif_col, "N/A")
            formula = row.get("Reduced Formula", "Unknown")
            print(f"\n      Sample CIF {i+1}: {formula}")
            cif_lines = cif.split('\n')[:30] if isinstance(cif, str) else ["N/A"]
            for line in cif_lines:
                print(f"      {line}")
            total_lines = len(cif.split('\n')) if isinstance(cif, str) else 0
            if total_lines > 30:
                print(f"      ... ({total_lines} lines total)")
        print()

    def _show_validity_stats(self, parquet_path: str):
        """Display validity statistics from processed parquet."""
        local_path = self._get_local_path(parquet_path)
        if not os.path.exists(local_path):
            print(f"      [verbose] File not found: {local_path}")
            return

        df = pd.read_parquet(local_path)

        print(f"\n      {'='*60}")
        print("      VALIDITY STATISTICS")
        print(f"      {'='*60}")
        print(f"      Total structures: {len(df)}")

        if "is_valid" in df.columns:
            valid_count = df["is_valid"].sum()
            print(f"      Valid: {valid_count} ({100*valid_count/len(df):.1f}%)")
        if "is_unique" in df.columns:
            unique_count = df["is_unique"].sum()
            print(f"      Unique: {unique_count} ({100*unique_count/len(df):.1f}%)")
        if "is_novel" in df.columns:
            novel_count = df["is_novel"].sum()
            print(f"      Novel: {novel_count} ({100*novel_count/len(df):.1f}%)")
        print()

    def _show_vun_metrics(self, csv_path: str):
        """Display VUN metrics from CSV output."""
        local_path = self._get_local_path(csv_path)
        if not os.path.exists(local_path):
            print(f"      [verbose] VUN CSV not found: {local_path}")
            return

        df = pd.read_csv(local_path)

        print(f"\n      {'='*60}")
        print("      VUN METRICS SUMMARY")
        print(f"      {'='*60}")
        print(df.to_string(index=False))
        print()

    def _show_ehull_stats(self, parquet_path: str):
        """Display E-hull statistics from parquet file."""
        local_path = self._get_local_path(parquet_path)
        if not os.path.exists(local_path):
            print(f"      [verbose] E-hull file not found: {local_path}")
            return

        df = pd.read_parquet(local_path)

        print(f"\n      {'='*60}")
        print("      E-HULL STATISTICS")
        print(f"      {'='*60}")

        ehull_col = "ehull_mace_mp" if "ehull_mace_mp" in df.columns else None
        if ehull_col:
            ehull_vals = df[ehull_col].dropna()
            print(f"      Total structures: {len(df)}")
            print(f"      Structures with E-hull: {len(ehull_vals)}")
            if len(ehull_vals) > 0:
                print(f"      E-hull range: {ehull_vals.min():.4f} to {ehull_vals.max():.4f} eV/atom")
                print(f"      E-hull mean: {ehull_vals.mean():.4f} eV/atom")
                print(f"      E-hull median: {ehull_vals.median():.4f} eV/atom")
                stable_count = (ehull_vals < 0.1).sum()
                print(f"      Stable (E-hull < 0.1 eV/atom): {stable_count} ({100*stable_count/len(ehull_vals):.1f}%)")

            print("\n      Sample E-hull values:")
            for _, row in df.head(5).iterrows():
                formula = row.get("Reduced Formula", row.get("formula", "Unknown"))
                ehull = row.get(ehull_col, "N/A")
                print(f"        {formula}: {ehull:.4f} eV/atom" if isinstance(ehull, float) else f"        {formula}: {ehull}")
        else:
            print(f"      Available columns: {list(df.columns)}")
            print("      No E-hull column found")
        print()

    def wait_for_job(self, job_id: str, timeout: int = 300, job_name: str = "") -> dict:
        """Wait for a job to complete."""
        print(f"      Waiting for job {job_name or job_id}...", end="", flush=True)
        start = time.time()
        last_status = ""
        first_not_found_at = None

        while time.time() - start < timeout:
            response = self.client.get(f"/jobs/{job_id}")
            if response.status_code != 200:
                payload = response.json() if hasattr(response, "json") else {}
                detail = payload.get("detail") if isinstance(payload, dict) else payload

                if response.status_code == 404 and detail == "Job not found":
                    if first_not_found_at is None:
                        first_not_found_at = time.time()
                    elapsed_not_found = time.time() - first_not_found_at
                    print(" [job-not-found]", end="", flush=True)
                    if elapsed_not_found < 20:
                        time.sleep(2)
                        continue
                    raise RuntimeError(
                        f"Job {job_id} not found for {elapsed_not_found:.1f}s. "
                        "In dev mode this usually means uvicorn autoreload restarted the API "
                        "and in-memory job state was reset."
                    )

                raise RuntimeError(f"Unexpected /jobs response: {response.status_code} payload={detail}")

            job = response.json()
            if not isinstance(job, dict) or "status" not in job:
                raise RuntimeError(f"Malformed /jobs payload for {job_id}: {job}")

            if job["status"] != last_status:
                print(f" [{job['status']}]", end="", flush=True)
                last_status = job["status"]
            if job["status"] in ["completed", "failed"]:
                elapsed = time.time() - start
                print(f" ({elapsed:.1f}s)")
                if job["status"] == "failed":
                    log_file = job.get("log_file")
                    if log_file:
                        print(f"      Job log file: {log_file}")
                        local_log = self._get_local_path(log_file)
                        if os.path.exists(local_log):
                            print(f"      Local log path: {local_log}")
                return job
            time.sleep(3)

        raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")
