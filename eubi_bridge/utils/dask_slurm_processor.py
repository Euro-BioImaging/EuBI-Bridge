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
            self.cluster.scale(optimal_workers)
            console.print(f"[blue]📊 Scaling to {optimal_workers} workers...[/blue]")

            # Connect client
            self.client = Client(self.cluster)
            console.print(f"[green]✅ Connected to cluster: {self.client.dashboard_link}[/green]")

            # Wait for workers to be ready
            self._wait_for_workers(optimal_workers)

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

        # Base configuration that works with most SLURM clusters
        config = {
            "queue": "htc-el8",  # Will be auto-detected if not specified
            "cores": 4,  # CPUs per worker
            "memory": "16GB",  # Memory per worker
            "walltime": "04:00:00",  # 4 hour limit
            "job_extra": [
                "--ntasks=1",
                "--cpus-per-task=4"
            ],
            "env_extra": [
                "export PYTHONPATH=$PYTHONPATH:{}".format(os.getcwd()),
                "module load python/3.11 2>/dev/null || module load python3 2>/dev/null || true"
            ],
            "python": "python3"
        }

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
            file_size_gb = Path(input_tiff).stat().st_size / (1024 ** 3)
            return file_size_gb
        except:
            return 1.0  # Default fallback

    def _wait_for_workers(self, expected_workers: int, timeout: int = 300):
        """Wait for workers to become available."""
        console.print("[blue]⏳ Waiting for workers to connect...[/blue]")

        start_time = time.time()
        while len(self.client.scheduler_info()['workers']) < expected_workers:
            if time.time() - start_time > timeout:
                current_workers = len(self.client.scheduler_info()['workers'])
                console.print(
                    f"[yellow]⚠️ Timeout waiting for workers. Got {current_workers}/{expected_workers}[/yellow]")
                break
            time.sleep(5)

        current_workers = len(self.client.scheduler_info()['workers'])
        console.print(f"[green]✅ {current_workers} workers ready[/green]")

    def _run_distributed_conversion(self, input_tiff: str, output_zarr_dir: str, **kwargs) -> bool:
        """Run the actual conversion using dask distributed processing."""
        try:
            console.print("[blue]🔄 Starting distributed conversion...[/blue]")

            # Create distributed tasks for conversion
            conversion_future = self._create_conversion_task(input_tiff, output_zarr_dir, **kwargs)

            # Wait for completion with progress monitoring
            result = conversion_future.result()

            if result:
                console.print("[green]✅ Distributed conversion completed successfully[/green]")
                return True
            else:
                console.print("[red]❌ Distributed conversion failed[/red]")
                return False

        except Exception as e:
            console.print(f"[red]❌ Conversion error: {e}[/red]")
            return False

    def _create_conversion_task(self, input_tiff: str, output_zarr_dir: str, **kwargs):
        """Create dask delayed task for conversion."""

        @delayed
        def distributed_convert():
            """Distributed conversion function."""
            # Import the original conversion function
            import sys
            import os
            sys.path.insert(0, os.getcwd())

            from core.converter import HighPerformanceConverter

            # Initialize converter with distributed-friendly settings
            converter = HighPerformanceConverter()

            # Run conversion
            return converter.convert(
                input_tiff=input_tiff,
                output_zarr_dir=output_zarr_dir,
                use_distributed=True,  # Signal that we're in distributed mode
                **kwargs
            )

        # Submit task to cluster
        return self.client.compute(distributed_convert(), sync=False)

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


def process_with_dask_slurm(input_tiff: str, output_zarr_dir: str, **kwargs) -> bool:
    """Convenience function for dask-jobqueue SLURM processing."""
    processor = DaskSlurmProcessor()
    return processor.process_conversion(input_tiff, output_zarr_dir, **kwargs)