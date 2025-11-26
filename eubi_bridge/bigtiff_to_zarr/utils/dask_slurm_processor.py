"""
Dask-based SLURM distributed processing for BigTIFF to OME-NGFF conversion.
Uses dask-jobqueue for robust HPC cluster integration.
"""

import os
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import tifffile
import zarr
from rich.console import Console

try:
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client, as_completed, Future
    import dask.array as da
    from dask import delayed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    print("Warning: dask-jobqueue not available. SLURM processing disabled.")

console = Console()


def _setup_worker_environment_func():
    """Setup environment on each worker (standalone function for pickling)."""
    import sys
    import os

    # Add project root to Python path
    project_root = os.environ.get('PROJECT_ROOT')
    if not project_root:
        # Try to find project root by looking for main.py or app.py
        cwd = os.getcwd()
        if os.path.exists(os.path.join(cwd, 'main.py')) or os.path.exists(os.path.join(cwd, 'app.py')):
            project_root = cwd
        else:
            # Fallback to current working directory
            project_root = cwd

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Change to project directory
    try:
        os.chdir(project_root)
    except:
        pass

    return f"Worker setup complete on {os.uname().nodename}, project_root: {project_root}"


def _test_imports_func():
    """Test importing core modules on each worker (standalone function for pickling)."""
    import sys

    try:
        from eubi_bridge.bigtiff_to_zarr.cli import convert_bigtiff_to_omezarr
        import tifffile
        import zarr
        import numpy
        return "All imports successful"
    except ImportError as e:
        return f"Import failed: {e}, sys.path: {sys.path[:3]}"


class DaskSlurmProcessor:
    """Robust SLURM distributed processing using dask-jobqueue."""

    def __init__(self):
        self.cluster = None
        self.client = None
        self.temp_dir = None

    def process_conversion(self, input_tiff: str, output_zarr_dir: str, **kwargs) -> bool:
        """Process BigTIFF conversion using dask-jobqueue SLURM cluster."""
        if not DASK_AVAILABLE:
            console.print("[red]❌ dask-jobqueue not available. Install with: pip install dask-jobqueue[/red]")
            return False

        try:
            console.print("[blue]🚀 Initializing Dask SLURM cluster...[/blue]")

            # Setup cluster configuration
            cluster_config = self._create_cluster_config(**kwargs)

            # Create SLURM cluster
            self.cluster = SLURMCluster(**cluster_config)
            console.print(f"[green]✅ SLURM cluster created[/green]")

            # Scale cluster based on data size
            optimal_workers = self._calculate_optimal_workers(input_tiff, **kwargs)
            console.print(f"[blue]📊 Scaling to {optimal_workers} workers...[/blue]")

            # Start with fewer workers initially to avoid overwhelming the scheduler
            initial_workers = min(4, optimal_workers)
            self.cluster.scale(initial_workers)
            console.print(f"[blue]🔄 Starting with {initial_workers} workers, will scale up as needed[/blue]")

            # Connect client
            self.client = Client(self.cluster.scheduler_address)
            console.print(f"[green]✅ Connected to cluster: {self.client.dashboard_link}[/green]")

            # Wait for workers to be ready
            workers_ready = self._wait_for_workers(initial_workers)
            if not workers_ready:
                console.print("[yellow]⚠️ Insufficient workers, falling back to single-node processing[/yellow]")
                return False

            # Scale up to optimal workers if we got initial workers
            current_workers = len(self.client.scheduler_info()['workers'])
            if current_workers > 0 and optimal_workers > initial_workers:
                console.print(f"[blue]📈 Scaling up from {current_workers} to {optimal_workers} workers...[/blue]")
                self.cluster.scale(optimal_workers)
                # Don't wait for all workers, proceed with what we have

            # Process conversion using dask
            success = self._run_distributed_conversion(input_tiff, output_zarr_dir, **kwargs)

            return success

        except Exception as e:
            console.print(f"[red]❌ Dask SLURM processing failed: {e}[/red]")
            return False

        finally:
            self._cleanup()

    def _create_cluster_config(self, **kwargs) -> Dict[str, Any]:
        """Create dask-jobqueue SLURM cluster configuration."""

        # Auto-detect memory and CPU requirements
        input_size_gb = self._estimate_input_size(kwargs.get("input_tiff", ""))

        # Base configuration optimized for quick startup
        config = {
            "queue": "htc-el8",
            "cores": 2,  # Start with fewer cores for faster allocation
            "memory": "8GB",  # Start with less memory for faster allocation
            "walltime": "01:00:00",  # Shorter time for higher priority
            "job_extra": [
                "--ntasks=1",
                "--cpus-per-task=2",
                "--mem=8G"
            ],
            "env_extra": [
                "export PYTHONPATH={}:$PYTHONPATH".format(os.getcwd()),
                "export PROJECT_ROOT={}".format(os.getcwd()),
                "cd {}".format(os.getcwd()),
                "module load python/3.11 2>/dev/null || module load python3 2>/dev/null || echo 'No module system'",
                "pip install --user --quiet tifffile zarr numpy numcodecs blosc2 2>/dev/null || true"
            ],
            "python": "python3",
            "log_directory": "/tmp/dask-slurm-logs",
            "death_timeout": 60,  # Kill workers if they don't start within 60s
            "job_script_prologue": [
                "mkdir -p /tmp/dask-slurm-logs",
                "echo 'Dask worker starting on node:' $(hostname)",
                "echo 'Available Python:' $(which python3)",
                "echo 'Python version:' $(python3 --version)",
                "echo 'Current directory:' $(pwd)",
                "echo 'PYTHONPATH:' $PYTHONPATH",
                "cd {}".format(os.getcwd()),
                "export PYTHONPATH={}:$PYTHONPATH".format(os.getcwd())
            ]
        }

        # Try to auto-detect the queue/partition if htc-el8 fails
        if not self._check_partition_exists("htc-el8"):
            detected_partition = self._auto_detect_partition()
            if detected_partition:
                config["queue"] = detected_partition
                console.print(f"[blue]📊 Using detected partition: {detected_partition}[/blue]")

        # Adjust resources based on input size
        if input_size_gb > 50:  # Large files need more resources
            config["memory"] = "32GB"
            config["cores"] = 8
            config["job_extra"].append("--cpus-per-task=8")

        # User can override specific settings
        if "slurm_queue" in kwargs:
            config["queue"] = kwargs["slurm_queue"]
        if "slurm_memory" in kwargs:
            config["memory"] = kwargs["slurm_memory"]
        if "slurm_cores" in kwargs:
            config["cores"] = kwargs["slurm_cores"]

        return config

    def _calculate_optimal_workers(self, input_tiff: str, **kwargs) -> int:
        """Calculate optimal number of workers based on data size."""
        try:
            input_size_gb = self._estimate_input_size(input_tiff)

            # Rule: 1 worker per 5-10GB of data, minimum 2, maximum 16
            optimal_workers = max(2, min(16, int(input_size_gb / 7)))

            console.print(f"[blue]📊 Input size: {input_size_gb:.1f}GB → {optimal_workers} workers[/blue]")
            return optimal_workers

        except Exception as e:
            console.print(f"[yellow]⚠️ Could not estimate size: {e}. Using 4 workers.[/yellow]")
            return 4

    def _estimate_input_size(self, input_tiff: str) -> float:
        """Estimate input file size in GB."""
        try:
            file_size_gb = Path(input_tiff).stat().st_size / (1024**3)
            return file_size_gb
        except:
            return 1.0  # Default fallback

    def _wait_for_workers(self, expected_workers: int, timeout: int = 180):
        """Wait for workers to become available with better diagnostics."""
        console.print("[blue]⏳ Waiting for workers to connect...[/blue]")

        start_time = time.time()
        last_count = 0

        while len(self.client.scheduler_info()['workers']) < expected_workers:
            current_workers = len(self.client.scheduler_info()['workers'])
            elapsed = time.time() - start_time

            # Show progress updates
            if current_workers != last_count:
                console.print(f"[blue]📊 Workers connected: {current_workers}/{expected_workers} (elapsed: {elapsed:.0f}s)[/blue]")
                last_count = current_workers

            # Check for timeout
            if elapsed > timeout:
                console.print(f"[yellow]⚠️ Timeout after {timeout}s. Got {current_workers}/{expected_workers} workers[/yellow]")

                # Show SLURM job status for debugging
                self._show_slurm_job_status()

                # Continue with available workers if we have at least 1
                if current_workers >= 1:
                    console.print(f"[yellow]🔄 Proceeding with {current_workers} available workers[/yellow]")
                    break
                else:
                    console.print("[red]❌ No workers available. Falling back to single-node processing.[/red]")
                    return False

            time.sleep(10)  # Check every 10 seconds

        current_workers = len(self.client.scheduler_info()['workers'])
        console.print(f"[green]✅ {current_workers} workers ready[/green]")
        return True

    def _run_distributed_conversion(self, input_tiff: str, output_zarr_dir: str, **kwargs) -> bool:
        """Run the actual conversion using dask distributed processing."""
        try:
            console.print("[blue]🔄 Starting distributed conversion...[/blue]")

            # Set up the distributed environment properly
            setup_results = self.client.run(_setup_worker_environment_func)
            console.print(f"[blue]📊 Worker setup results: {len(setup_results)} workers configured[/blue]")

            # Test imports on all workers
            import_results = self.client.run(_test_imports_func)
            console.print(f"[blue]📊 Import test results: {import_results}[/blue]")

            # Create distributed tasks for conversion
            conversion_future = self._create_conversion_task(input_tiff, output_zarr_dir, **kwargs)

            # Wait for completion with progress monitoring
            result = conversion_future.result(timeout=None)

            if result:
                console.print("[green]✅ Distributed conversion completed successfully[/green]")
                return True
            else:
                console.print("[red]❌ Distributed conversion failed[/red]")
                return False

        except Exception as e:
            console.print(f"[red]❌ Conversion error: {e}[/red]")
            import traceback
            console.print(f"[red]Full traceback: {traceback.format_exc()}[/red]")
            return False

    def _create_conversion_task(self, input_tiff: str, output_zarr_dir: str, **kwargs):
        """Create dask delayed task for conversion."""

        # Create standalone conversion function for pickling
        def distributed_convert_func():
            """Distributed conversion function (standalone for pickling)."""
            # Ensure proper environment setup
            import sys
            import os

            # Add project root to path
            project_root = os.environ.get('PROJECT_ROOT')
            if not project_root:
                project_root = os.getcwd()

            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            try:
                os.chdir(project_root)
            except:
                pass

            # Import the standalone conversion function from cli
            import sys
            sys.path.insert(0, os.getcwd())
            from eubi_bridge.bigtiff_to_zarr.cli import convert_bigtiff_to_omezarr

            # Run conversion with distributed disabled to prevent recursive calls
            kwargs_copy = kwargs.copy()
            kwargs_copy['on_slurm'] = False  # Disable distributed processing within workers

            # Run conversion using the standalone function
            success = convert_bigtiff_to_omezarr(
                input_tiff=input_tiff,
                output_zarr_dir=output_zarr_dir,
                **kwargs_copy
            )
            return success

        # Create delayed version
        delayed_convert = delayed(distributed_convert_func)

        # Submit task to cluster
        future = self.client.compute(delayed_convert(), sync=False)
        return future

    def _cleanup(self):
        """Clean up dask cluster and resources."""
        try:
            if self.client:
                self.client.close()
                console.print("[blue]📋 Client closed[/blue]")

            if self.cluster:
                self.cluster.close()
                console.print("[blue]📋 Cluster closed[/blue]")

        except Exception as e:
            console.print(f"[yellow]⚠️ Cleanup warning: {e}[/yellow]")

    def _show_slurm_job_status(self):
        """Show SLURM job status for debugging."""
        try:
            import subprocess
            result = subprocess.run(["squeue", "--user", os.environ.get("USER", "unknown"), "--format=%i,%T,%R"],
                                   capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                console.print("[blue]📋 Current SLURM jobs:[/blue]")
                lines = result.stdout.strip().split('\n')
                for line in lines[1:]:  # Skip header
                    if line.strip():
                        job_id, status, reason = line.split(',')
                        console.print(f"[blue]  Job {job_id}: {status} ({reason})[/blue]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Could not check SLURM status: {e}[/yellow]")

    def _check_partition_exists(self, partition: str) -> bool:
        """Check if a SLURM partition exists."""
        try:
            import subprocess
            result = subprocess.run(["sinfo", "-p", partition], capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except:
            return False

    def _auto_detect_partition(self) -> Optional[str]:
        """Auto-detect available SLURM partition."""
        try:
            import subprocess
            result = subprocess.run(["sinfo", "--format=%P,%S", "--noheader"],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                partitions = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        partition = line.split(',')[0].strip()
                        if partition and not partition.endswith('*'):
                            partitions.append(partition)

                # Prefer common HPC partition names
                preferred = ['gpu', 'compute', 'cpu', 'main', 'standard', 'normal']
                for pref in preferred:
                    if pref in partitions:
                        return pref

                # Return first available partition
                return partitions[0] if partitions else None
        except:
            return None




def process_with_dask_slurm(input_tiff: str, output_zarr_dir: str, **kwargs) -> bool:
    """Convenience function for dask-jobqueue SLURM processing."""
    processor = DaskSlurmProcessor()
    return processor.process_conversion(input_tiff, output_zarr_dir, **kwargs)