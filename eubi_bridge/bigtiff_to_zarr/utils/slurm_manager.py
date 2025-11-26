"""
SLURM Job Manager for Distributed BigTIFF to OME-NGFF Conversion
Handles job submission, coordination, and resource management across HPC clusters.
"""

import os
import subprocess
import tempfile
import json
import time
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import tifffile
import zarr

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

console = Console()


class SlurmJobManager:
    """Manages SLURM job submission and coordination for distributed processing."""

    def __init__(self):
        self.job_ids = []
        self.temp_dir = None
        self.work_dir = None
        self.available_partition = None

    def submit_conversion_job(self, **kwargs) -> bool:
        """Submit and coordinate distributed conversion across SLURM cluster."""
        try:
            console.print("[blue]🚀 Initializing SLURM distributed processing...[/blue]")

            # Validate SLURM environment
            if not self._check_slurm_available():
                console.print("[red]❌ SLURM not available. Falling back to single-node processing.[/red]")
                return self._fallback_to_single_node(**kwargs)

            # Create working directory for distributed processing
            self.temp_dir = tempfile.mkdtemp(prefix="bigtiff_slurm_")
            self.work_dir = Path(self.temp_dir)

            console.print(f"[blue]📁 Created work directory: {self.work_dir}[/blue]")

            # Analyze input file and create processing plan
            # input_tiff = kwargs["input_tiff"]
            processing_plan = self._analyze_input_and_create_plan(**kwargs)

            console.print(f"[green]📊 Analysis complete: {processing_plan['total_chunks']} chunks across {processing_plan['recommended_nodes']} nodes[/green]")

            # Submit coordinator job
            coordinator_job_id = self._submit_coordinator_job(processing_plan, **kwargs)

            # Submit worker jobs
            worker_job_ids = self._submit_worker_jobs(processing_plan, coordinator_job_id, **kwargs)

            self.job_ids = [coordinator_job_id] + worker_job_ids

            console.print(f"[green]✅ Submitted {len(self.job_ids)} jobs to SLURM cluster[/green]")
            console.print(f"[blue]📋 Coordinator Job: {coordinator_job_id}[/blue]")
            console.print(f"[blue]👥 Worker Jobs: {', '.join(worker_job_ids)}[/blue]")

            # Monitor job completion
            return self._monitor_job_completion(**kwargs)

        except Exception as e:
            console.print(f"[red]❌ SLURM processing failed: {e}[/red]")
            return self._fallback_to_single_node(**kwargs)

        finally:
            # Cleanup temporary files
            self._cleanup()

    def _check_slurm_available(self) -> bool:
        """Check if SLURM is available and accessible."""
        try:
            # Check if SLURM commands are available
            result = subprocess.run(["which", "sbatch"], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return False

            # Check if we can query SLURM info
            result = subprocess.run(["sinfo", "--version"], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return False

            console.print(f"[green]✅ SLURM detected: {result.stdout.strip()}[/green]")

            # Detect available partitions
            self.available_partition = self._detect_available_partition()
            if not self.available_partition:
                console.print("[yellow]⚠️ Warning: No accessible SLURM partitions found[/yellow]")
                return False

            console.print(f"[green]📊 Using partition: {self.available_partition}[/green]")
            return True

        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False

    def _detect_available_partition(self) -> Optional[str]:
        """Detect the first available SLURM partition for job submission."""
        try:
            # Get list of partitions and their states
            result = subprocess.run(
                ["sinfo", "-h", "-o", "%P %a"],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode != 0:
                return None

            # Parse partition information
            partitions = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        partition_name = parts[0].rstrip('*')  # Remove default partition marker
                        partition_state = parts[1]

                        # Prefer 'up' partitions, but accept others as fallback
                        if partition_state.lower() == 'up':
                            return partition_name
                        else:
                            partitions.append((partition_name, partition_state))

            # If no 'up' partition found, return the first available one
            if partitions:
                console.print(f"[yellow]⚠️ No 'up' partitions found, using: {partitions[0][0]} ({partitions[0][1]})[/yellow]")
                return partitions[0][0]

            return None

        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return None

    def _analyze_input_and_create_plan(self, input_tiff: str, **kwargs) -> Dict[str, Any]:
        """Analyze input file and create distributed processing plan."""
        console.print("[blue]🔍 Analyzing input file for distributed processing...[/blue]")

        # Get file information
        input_path = Path(input_tiff)
        file_size_gb = input_path.stat().st_size / (1024**3)

        # Analyze TIFF structure
        with tifffile.TiffFile(input_tiff) as tiff:
            page = tiff.pages[0]
            shape = page.shape
            dtype = page.dtype

            # Calculate total data size
            total_elements = int(np.prod(shape)) * len(tiff.pages)
            total_size_gb = total_elements * np.dtype(dtype).itemsize / (1024**3)

        # Determine optimal chunking strategy for distributed processing
        chunk_sizes = {
            "z": kwargs.get("z_chunk", 96),
            "y": kwargs.get("y_chunk", 96),
            "x": kwargs.get("x_chunk", 96)
        }

        # Calculate number of chunks
        if len(shape) == 2:
            chunks_per_dim = [
                max(1, int(shape[0]) // chunk_sizes["y"]),
                max(1, int(shape[1]) // chunk_sizes["x"])
            ]
            total_chunks = int(np.prod(chunks_per_dim) * len(tiff.pages))
        else:
            chunks_per_dim = [
                max(1, int(shape[0]) // chunk_sizes["z"]),
                max(1, int(shape[1]) // chunk_sizes["y"]),
                max(1, int(shape[2]) // chunk_sizes["x"])
            ]
            total_chunks = int(np.prod(chunks_per_dim))

        # Determine optimal number of nodes based on data size and chunk count
        # Rule: 1 node per 10GB of data, minimum 2 nodes, maximum based on chunks
        recommended_nodes = max(2, min(int(total_size_gb / 10), total_chunks, 16))
        chunks_per_node = max(1, total_chunks // recommended_nodes)

        processing_plan = {
            "file_size_gb": float(file_size_gb),
            "total_size_gb": float(total_size_gb),
            "shape": [int(dim) for dim in shape],  # Convert to regular Python list of ints
            "dtype": str(dtype),
            "total_chunks": int(total_chunks),
            "chunks_per_dim": [int(chunk) for chunk in chunks_per_dim],  # Convert to regular Python list
            "chunks_per_node": int(chunks_per_node),
            "recommended_nodes": int(recommended_nodes),
            "chunk_sizes": chunk_sizes,
            "estimated_memory_per_node_gb": int(max(8, int(total_size_gb / recommended_nodes * 1.5))),
            "estimated_runtime_hours": int(max(1, int(total_size_gb / (recommended_nodes * 2))))  # ~2GB/hour per node
        }

        # Save processing plan to work directory
        plan_file = self.work_dir / "processing_plan.json"
        with open(plan_file, 'w') as f:
            json.dump(processing_plan, f, indent=2)

        return processing_plan

    def _submit_coordinator_job(self, processing_plan: Dict[str, Any], **kwargs) -> str:
        """Submit coordinator job that manages the distributed conversion."""
        console.print("[blue]📋 Submitting coordinator job...[/blue]")

        # Create coordinator script
        coordinator_script = self._create_coordinator_script(processing_plan, **kwargs)
        script_path = self.work_dir / "coordinator_job.sh"

        with open(script_path, 'w') as f:
            f.write(coordinator_script)

        os.chmod(script_path, 0o755)

        # Submit coordinator job
        sbatch_cmd = [
            "sbatch",
            "--job-name=bigtiff-coordinator",
            f"--output={self.work_dir}/coordinator.out",
            f"--error={self.work_dir}/coordinator.err",
            "--ntasks=1",
            f"--mem={max(8, processing_plan['estimated_memory_per_node_gb'])}G",
            f"--time={processing_plan['estimated_runtime_hours']:02d}:00:00",
            f"--partition={self.available_partition}",
            str(script_path)
        ]

        result = subprocess.run(sbatch_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to submit coordinator job: {result.stderr}")

        # Extract job ID
        job_id = result.stdout.strip().split()[-1]
        console.print(f"[green]✅ Coordinator job submitted: {job_id}[/green]")

        return job_id

    def _submit_worker_jobs(self, processing_plan: Dict[str, Any], coordinator_job_id: str, **kwargs) -> List[str]:
        """Submit worker jobs for distributed chunk processing."""
        console.print(f"[blue]👥 Submitting {processing_plan['recommended_nodes']} worker jobs...[/blue]")

        worker_job_ids = []

        for worker_id in range(processing_plan["recommended_nodes"]):
            # Create worker script
            worker_script = self._create_worker_script(worker_id, processing_plan, **kwargs)
            script_path = self.work_dir / f"worker_{worker_id}.sh"

            with open(script_path, 'w') as f:
                f.write(worker_script)

            os.chmod(script_path, 0o755)

            # Submit worker job with dependency on coordinator
            sbatch_cmd = [
                "sbatch",
                f"--job-name=bigtiff-worker-{worker_id}",
                f"--output={self.work_dir}/worker_{worker_id}.out",
                f"--error={self.work_dir}/worker_{worker_id}.err",
                "--ntasks=1",
                f"--cpus-per-task={max(4, min(16, processing_plan['chunks_per_node']))}",
                f"--mem={processing_plan['estimated_memory_per_node_gb']}G",
                f"--time={processing_plan['estimated_runtime_hours']:02d}:00:00",
                f"--dependency=after:{coordinator_job_id}",
                f"--partition={self.available_partition}",
                str(script_path)
            ]

            result = subprocess.run(sbatch_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                console.print(f"[yellow]⚠️ Failed to submit worker {worker_id}: {result.stderr}[/yellow]")
                continue

            # Extract job ID
            job_id = result.stdout.strip().split()[-1]
            worker_job_ids.append(job_id)

        console.print(f"[green]✅ Submitted {len(worker_job_ids)} worker jobs[/green]")
        return worker_job_ids

    def _create_coordinator_script(self, processing_plan: Dict[str, Any], **kwargs) -> str:
        """Create SLURM script for coordinator job."""
        script = f"""#!/bin/bash
#SBATCH --job-name=bigtiff-coordinator

set -e

echo "=== BigTIFF to OME-NGFF Coordinator Job Started ==="
echo "Coordinator starting at $(date)"
echo "Working directory: {self.work_dir}"
echo "Processing plan: {processing_plan['total_chunks']} chunks across {processing_plan['recommended_nodes']} nodes"

# Load required modules (adjust for your HPC environment)
# module load python/3.11
# module load mpi/openmpi

cd {self.work_dir}

# Initialize output zarr store
python3 << 'EOF'
import sys
sys.path.append('{os.getcwd()}')
from eubi_bridge.bigtiff_to_zarr.utils.slurm_coordinator import initialize_zarr_store
initialize_zarr_store('{kwargs["input_tiff"]}', '{kwargs["output_zarr_dir"]}', {json.dumps(kwargs)})
EOF

echo "✅ Zarr store initialized"

# Create task queue for workers
python3 << 'EOF'
import sys
sys.path.append('{os.getcwd()}')
from eubi_bridge.bigtiff_to_zarr.utils.slurm_coordinator import create_task_queue
create_task_queue('{self.work_dir}', {json.dumps(processing_plan)})
EOF

echo "✅ Task queue created"
echo "Coordinator completed at $(date)"
"""
        return script

    def _create_worker_script(self, worker_id: int, processing_plan: Dict[str, Any], **kwargs) -> str:
        """Create SLURM script for worker job."""
        script = f"""#!/bin/bash
#SBATCH --job-name=bigtiff-worker-{worker_id}

set -e

echo "=== BigTIFF Worker {worker_id} Started ==="
echo "Worker {worker_id} starting at $(date)"
echo "Working directory: {self.work_dir}"

# Load required modules
# module load python/3.11

cd {self.work_dir}

# Process assigned chunks
python3 << 'EOF'
import sys
sys.path.append('{os.getcwd()}')
from eubi_bridge.bigtiff_to_zarr.utils.slurm_worker import process_worker_chunks
process_worker_chunks({worker_id}, '{self.work_dir}', '{kwargs["input_tiff"]}', '{kwargs["output_zarr_dir"]}', {json.dumps(kwargs)})
EOF

echo "✅ Worker {worker_id} completed at $(date)"
"""
        return script

    def _monitor_job_completion(self, **kwargs) -> bool:
        """Monitor SLURM jobs until completion."""
        console.print("[blue]📊 Monitoring job progress...[/blue]")

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:

            task = progress.add_task("SLURM Jobs", total=len(self.job_ids))

            completed_jobs = set()
            start_time = time.time()

            while len(completed_jobs) < len(self.job_ids):
                # Check job statuses
                for job_id in self.job_ids:
                    if job_id in completed_jobs:
                        continue

                    status = self._get_job_status(job_id)
                    if status in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"]:
                        completed_jobs.add(job_id)
                        progress.update(task, advance=1)

                        if status == "FAILED":
                            console.print(f"[red]❌ Job {job_id} failed[/red]")
                        elif status == "COMPLETED":
                            console.print(f"[green]✅ Job {job_id} completed[/green]")

                # Check for timeout (4 hours max)
                if time.time() - start_time > 14400:
                    console.print("[red]⏰ Job monitoring timeout reached[/red]")
                    return False

                # Sleep between checks
                time.sleep(30)

        # Verify output and collect results
        return self._verify_and_finalize_output(**kwargs)

    def _get_job_status(self, job_id: str) -> str:
        """Get SLURM job status."""
        try:
            result = subprocess.run(
                ["squeue", "-j", job_id, "--noheader", "--format=%T"],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            else:
                # Job might be completed and not in queue, check sacct
                result = subprocess.run(
                    ["sacct", "-j", job_id, "--noheader", "--format=State", "-X"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip().split()[0]

            return "UNKNOWN"

        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            return "UNKNOWN"

    def _verify_and_finalize_output(self, **kwargs) -> bool:
        """Verify distributed processing output and finalize zarr store."""
        console.print("[blue]🔍 Verifying distributed processing results...[/blue]")

        try:
            output_path = Path(kwargs["output_zarr_dir"])

            # Check if output exists and has expected structure
            if not output_path.exists():
                console.print("[red]❌ Output directory not found[/red]")
                return False

            # Verify zarr store integrity
            store = zarr.open(str(output_path), mode='r')
            console.print(f"[green]✅ Output verified: {list(store.keys())}[/green]")

            # Show completion summary
            self._show_completion_summary(**kwargs)

            return True

        except Exception as e:
            console.print(f"[red]❌ Output verification failed: {e}[/red]")
            return False

    def _show_completion_summary(self, **kwargs):
        """Display completion summary."""
        table = Table(title="SLURM Distributed Conversion Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Jobs Submitted", str(len(self.job_ids)))
        table.add_row("Coordinator Job", self.job_ids[0] if self.job_ids else "N/A")
        table.add_row("Worker Jobs", str(len(self.job_ids) - 1))
        table.add_row("Output Location", kwargs["output_zarr_dir"])
        table.add_row("Zarr Format", f"v{kwargs.get('zarr_format', 2)}")

        console.print(table)

    def _fallback_to_single_node(self, **kwargs) -> bool:
        """Fallback to single-node processing."""
        console.print("[yellow]⚠️ Falling back to single-node processing...[/yellow]")

        # Import and call standard conversion
        import asyncio
        from eubi_bridge.bigtiff_to_zarr.cli import _convert_async

        # Remove on_slurm parameter to avoid recursion
        fallback_kwargs = kwargs.copy()
        fallback_kwargs.pop("on_slurm", None)

        return asyncio.run(_convert_async(**fallback_kwargs))

    def _cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and Path(self.temp_dir).exists():
            try:
                shutil.rmtree(self.temp_dir)
                console.print(f"[blue]🧹 Cleaned up temporary directory: {self.temp_dir}[/blue]")
            except Exception as e:
                console.print(f"[yellow]⚠️ Failed to cleanup {self.temp_dir}: {e}[/yellow]")