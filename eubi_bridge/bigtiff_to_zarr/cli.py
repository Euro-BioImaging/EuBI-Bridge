#!/usr/bin/env python3
"""
Command Line Interface for High-Performance BigTIFF to OME-NGFF Converter
Provides both simple function interface and full CLI with advanced options.
"""

import argparse
import asyncio
import sys
import json
import time
from pathlib import Path
from typing import Optional, Union, Dict, Any

import numpy as np
import psutil
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel

from .core.converter import HighPerformanceConverter
from .core.progress_monitor import ProgressMonitor
from .utils.hardware_detection import HardwareDetector
from .utils.config_profiles import ConfigProfileManager

console = Console()


def convert_bigtiff_to_omezarr(
        input_tiff,
        output_zarr_dir,
        dimension_order="tczyx",
        dtype=None,
        compressor="blosc",
        time_scale=1,
        channel_scale=1,
        z_scale=2,
        y_scale=2,
        x_scale=2,
        time_chunk=1,
        channel_chunk=1,
        z_chunk=96,
        y_chunk=96,
        x_chunk=96,
        min_dimension_size=64,
        n_layers=None,
        auto_chunk=False,
        overwrite=False,
        save_omexml=False,
        zarr_format=2,
        **kwargs
):
    """
    Convert BigTIFF to OME-NGFF/OME-Zarr format with high-performance optimizations.

    Args:
        input_tiff: Path to input BigTIFF file
        output_zarr_dir: Path to output OME-NGFF directory
        dimension_order: Order of dimensions in the data (default: "tczyx")
        dtype: Output data type (None for auto-detection)
        compressor: Compression algorithm ("blosc", "blosc2-lz4", "blosc2-zstd", "lz4", "none")
        time_scale, channel_scale, z_scale, y_scale, x_scale: Scale factors for pyramid levels
        time_chunk, channel_chunk, z_chunk, y_chunk, x_chunk: Chunk sizes for each dimension
        min_dimension_size: Minimum size for pyramid levels
        n_layers: Number of pyramid layers (None for auto-calculation)
        auto_chunk: Automatically determine optimal chunk sizes
        overwrite: Overwrite existing output directory
        save_omexml: Save OME-XML metadata alongside Zarr
        zarr_format: Zarr format version (2 or 3)
        **kwargs: Additional configuration options

    Returns:
        bool: True if conversion successful, False otherwise
    """
    return asyncio.run(_convert_async(
        input_tiff=input_tiff,
        output_zarr_dir=output_zarr_dir,
        dimension_order=dimension_order,
        dtype=dtype,
        compressor=compressor,
        time_scale=time_scale,
        channel_scale=channel_scale,
        z_scale=z_scale,
        y_scale=y_scale,
        x_scale=x_scale,
        time_chunk=time_chunk,
        channel_chunk=channel_chunk,
        z_chunk=z_chunk,
        y_chunk=y_chunk,
        x_chunk=x_chunk,
        min_dimension_size=min_dimension_size,
        n_layers=n_layers,
        auto_chunk=auto_chunk,
        overwrite=overwrite,
        save_omexml=save_omexml,
        zarr_format=zarr_format,
        **kwargs
    ))


async def _convert_async(
        input_tiff,
        output_zarr_dir,
        dimension_order="tczyx",
        dtype=None,
        compressor="blosc",
        time_scale=1,
        channel_scale=1,
        z_scale=2,
        y_scale=2,
        x_scale=2,
        time_chunk=1,
        channel_chunk=1,
        z_chunk=96,
        y_chunk=96,
        x_chunk=96,
        min_dimension_size=64,
        n_layers=None,
        auto_chunk=False,
        overwrite=False,
        save_omexml=False,
        zarr_format=2,
        **kwargs
):
    """Async implementation of the conversion function."""

    # Validate input and output paths
    input_path = Path(input_tiff)
    output_path = Path(output_zarr_dir)

    if not input_path.exists():
        console.print(f"[red]Error: Input file '{input_path}' does not exist[/red]")
        return False

    if output_path.exists() and not overwrite:
        console.print(
            f"[red]Error: Output directory '{output_path}' already exists. Use overwrite=True to replace it.[/red]")
        return False

    # Map compression options
    compression_mapping = {
        "blosc": "blosc2-lz4",
        "blosc2-lz4": "blosc2-lz4",
        "blosc2-zstd": "blosc2-zstd",
        "lz4": "lz4",
        "none": "none"
    }

    # Build configuration
    config = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "dimension_order": dimension_order,
        "dtype": dtype,
        "compression": compression_mapping.get(compressor, "blosc2-lz4"),
        "compression_level": kwargs.get("compression_level", 3),
        "pyramid_scales": {
            "time": time_scale,
            "channel": channel_scale,
            "z": z_scale,
            "y": y_scale,
            "x": x_scale
        },
        "chunk_sizes": {
            "time": time_chunk,
            "channel": channel_chunk,
            "z": z_chunk,
            "y": y_chunk,
            "x": x_chunk
        },
        "min_dimension_size": min_dimension_size,
        "pyramid_levels": n_layers,
        "auto_chunk": auto_chunk,
        "overwrite": overwrite,
        "save_omexml": save_omexml,
        "zarr_format": zarr_format,
        **kwargs
    }

    # Auto-detect hardware and optimize configuration if not specified
    if auto_chunk or kwargs.get("auto_optimize", True):
        console.print("[blue]Detecting hardware configuration...[/blue]")
        hardware_detector = HardwareDetector()
        await hardware_detector.detect_hardware()

        profile_manager = ConfigProfileManager()
        profile_name = hardware_detector.recommend_profile()
        profile_config = profile_manager.get_profile_config(profile_name)

        # Only use profile defaults for unspecified values
        # Preserve explicitly passed parameters
        user_specified_chunks = (
                time_chunk != 1 or channel_chunk != 1 or z_chunk != 96 or
                y_chunk != 96 or x_chunk != 96
        )

        # Merge profile configuration but preserve user-specified parameters
        for key, value in profile_config.items():
            should_override = False

            # Only override if the value wasn't explicitly set by user
            if key not in config or config[key] is None:
                should_override = True
            elif key == "chunk_sizes" and not user_specified_chunks:
                # Only override chunk sizes if user didn't specify custom chunks
                should_override = True
            elif key in ["workers", "threads_per_worker"] and config.get(key) == "auto":
                # Handle "auto" values
                should_override = True

            if should_override:
                if key == "workers" and value == "auto":
                    config[key] = psutil.cpu_count()
                else:
                    config[key] = value

        console.print(f"[green]Using {profile_name.upper()} optimization profile[/green]")
        console.print(
            f"Using {config.get('workers', 'auto')} workers with {config.get('threads_per_worker', 2)} threads per worker.")

        # Show chunk configuration
        chunks = config.get("chunk_sizes", {})
        if chunks:
            chunk_info = ", ".join([f"{k}={v}" for k, v in chunks.items() if v > 0])
            console.print(f"Chunk sizes: {chunk_info}")
    else:
        # Manual configuration - still need basic hardware detection for workers
        console.print("[blue]Detecting hardware configuration...[/blue]")
        hardware_detector = HardwareDetector()
        await hardware_detector.detect_hardware()

        profile_manager = ConfigProfileManager()
        profile_name = hardware_detector.recommend_profile()

        # Only set workers if not specified
        if "workers" not in config or config["workers"] == "auto":
            config["workers"] = psutil.cpu_count()
        if "threads_per_worker" not in config or config["threads_per_worker"] is None:
            config["threads_per_worker"] = 2

        console.print(f"[green]Using {profile_name.upper()} optimization profile[/green]")
        console.print(
            f"Using {config.get('workers')} workers with {config.get('threads_per_worker')} threads per worker.")

        # Show user-specified chunk configuration
        chunks = config.get("chunk_sizes", {})
        if chunks:
            chunk_info = ", ".join([f"{k}={v}" for k, v in chunks.items() if v > 0])
            console.print(f"[yellow]User-specified chunk sizes: {chunk_info}[/yellow]")

    # Initialize converter
    converter = HighPerformanceConverter(config)

    # Create progress monitor for CLI output
    progress_monitor = CLIProgressMonitor()

    # Start conversion
    console.print(f"[blue]Starting conversion: {input_path} -> {output_path}[/blue]")

    start_time = time.time()
    try:
        success = await converter.convert(
            input_path=str(input_path),
            output_path=str(output_path),
            progress_monitor=progress_monitor
        )
    except Exception as e:
        console.print(f"[red]Conversion failed with error: {e}[/red]")
        import traceback
        console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        return False
    end_time = time.time()

    if success:
        duration = end_time - start_time
        console.print(f"[green]✓ Conversion completed successfully in {duration:.1f} seconds[/green]")

        # Display summary statistics
        stats = await converter.get_performance_stats()
        _display_conversion_summary(stats, duration)

    else:
        console.print(f"[red]✗ Conversion failed[/red]")

    return success


class CLIProgressMonitor(ProgressMonitor):
    """Progress monitor for CLI interface using Rich progress bars."""

    def __init__(self):
        super().__init__()
        self.progress = None
        self.task_id = None
        self.current_stage = ""

    def start_monitoring(self, total_items: int, description: str = "Converting"):
        """Start progress monitoring."""
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        )
        self.progress.start()
        self.task_id = self.progress.add_task(description, total=total_items)

    async def update_progress(self, level: int = None, chunks_completed: int = None,
                              total_chunks: int = None, bytes_processed: int = None):
        """Update progress."""
        if self.progress and self.task_id is not None:
            # Calculate progress based on available data
            if chunks_completed is not None and total_chunks is not None:
                progress_value = chunks_completed

                # Update description with current stage info
                description = f"Converting"
                if level is not None:
                    description += f" - Level {level}"
                if total_chunks > 0:
                    description += f" ({chunks_completed}/{total_chunks} chunks)"

                self.progress.update(self.task_id, completed=progress_value,
                                     total=total_chunks, description=description)

    async def initialize_conversion(self, analysis: Dict[str, Any]):
        """Initialize conversion progress."""
        total_levels = len(analysis.get("pyramid_levels", []))
        console.print(f"[blue]Processing {total_levels} pyramid levels...[/blue]")

        # Start progress monitoring with reasonable total
        total_chunks = sum(1 for _ in analysis.get("pyramid_levels", []))
        self.start_monitoring(total_chunks, "Converting pyramid levels")

    async def report_error(self, error_message: str):
        """Report an error."""
        console.print(f"[red]Error: {error_message}[/red]")

    async def report_warning(self, warning_message: str):
        """Report a warning."""
        console.print(f"[yellow]Warning: {warning_message}[/yellow]")

    def finish_monitoring(self):
        """Finish progress monitoring."""
        if self.progress:
            self.progress.stop()
            self.progress = None
            self.task_id = None


def _display_conversion_summary(stats: Dict[str, Any], duration: float):
    """Display conversion summary statistics."""
    table = Table(title="Conversion Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    # File size information
    if "input_size_mb" in stats:
        table.add_row("Input Size", f"{stats['input_size_mb']:.1f} MB")
    if "output_size_mb" in stats:
        table.add_row("Output Size", f"{stats['output_size_mb']:.1f} MB")
    if "compression_ratio" in stats:
        table.add_row("Compression Ratio", f"{stats['compression_ratio']:.2f}:1")

    # Performance metrics
    table.add_row("Duration", f"{duration:.1f} seconds")
    if "average_throughput_mb_s" in stats:
        table.add_row("Average Throughput", f"{stats['average_throughput_mb_s']:.1f} MB/s")
    if "peak_memory_usage_gb" in stats:
        table.add_row("Peak Memory Usage", f"{stats['peak_memory_usage_gb']:.1f} GB")

    # Processing details
    if "workers_used" in stats:
        table.add_row("Workers Used", str(stats['workers_used']))
    if "pyramid_levels" in stats:
        table.add_row("Pyramid Levels", str(stats['pyramid_levels']))

    console.print(table)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="High-Performance BigTIFF to OME-NGFF Converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python cli.py input.tif output.zarr

  # High-performance conversion with custom settings
  python cli.py input.tif output.zarr --compressor blosc2-zstd --workers 8 --auto-chunk

  # Convert with specific chunk sizes
  python cli.py input.tif output.zarr --z-chunk 64 --y-chunk 512 --x-chunk 512

  # Show system information
  python cli.py --system-info
        """
    )

    # Positional arguments
    parser.add_argument("input_tiff", nargs="?", help="Input BigTIFF file path")
    parser.add_argument("output_zarr", nargs="?", help="Output OME-NGFF directory path")

    # Conversion options
    parser.add_argument("--dimension-order", default="tczyx",
                        help="Dimension order (default: tczyx)")
    parser.add_argument("--dtype", help="Output data type (auto-detected if not specified)")
    parser.add_argument("--compressor", default="blosc",
                        choices=["blosc", "blosc2-lz4", "blosc2-zstd", "lz4", "none"],
                        help="Compression algorithm (default: blosc)")
    parser.add_argument("--compression-level", type=int, default=3,
                        help="Compression level 1-9 (default: 3)")

    # Pyramid options
    parser.add_argument("--time-scale", type=int, default=1, help="Time pyramid scale factor")
    parser.add_argument("--channel-scale", type=int, default=1, help="Channel pyramid scale factor")
    parser.add_argument("--z-scale", type=int, default=2, help="Z pyramid scale factor")
    parser.add_argument("--y-scale", type=int, default=2, help="Y pyramid scale factor")
    parser.add_argument("--x-scale", type=int, default=2, help="X pyramid scale factor")
    parser.add_argument("--n-layers", type=int, help="Number of pyramid layers (auto if not specified)")
    parser.add_argument("--min-dimension-size", type=int, default=64,
                        help="Minimum dimension size for pyramid levels")

    # Chunking options
    parser.add_argument("--time-chunk", type=int, default=1, help="Time chunk size")
    parser.add_argument("--channel-chunk", type=int, default=1, help="Channel chunk size")
    parser.add_argument("--z-chunk", type=int, default=96, help="Z chunk size")
    parser.add_argument("--y-chunk", type=int, default=96, help="Y chunk size")
    parser.add_argument("--x-chunk", type=int, default=96, help="X chunk size")
    parser.add_argument("--auto-chunk", action="store_true",
                        help="Automatically determine optimal chunk sizes")

    # Performance options
    parser.add_argument("--workers", type=int, help="Number of worker processes (auto-detected if not specified)")
    parser.add_argument("--chunk-memory-mb", type=int, help="Memory per chunk in MB")
    parser.add_argument("--use-numa", action="store_true", help="Enable NUMA optimization")
    parser.add_argument("--use-async-io", action="store_true", default=True, help="Enable async I/O")
    parser.add_argument("--profile", choices=["hpc", "workstation", "cloud", "constrained"],
                        help="Use predefined optimization profile")

    # Output options
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    parser.add_argument("--save-omexml", action="store_true", help="Save OME-XML metadata")
    parser.add_argument("--zarr-format", type=int, default=2, choices=[2, 3],
                        help="Zarr format version (default: 2)")

    # Information commands
    parser.add_argument("--system-info", action="store_true", help="Show system information and exit")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark and exit")
    parser.add_argument("--profiles", action="store_true", help="List available optimization profiles and exit")

    # Debug options
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--config-file", help="Load configuration from JSON file")

    args = parser.parse_args()

    # Handle information commands
    if args.system_info:
        asyncio.run(_show_system_info())
        return

    if args.benchmark:
        asyncio.run(_run_benchmark())
        return

    if args.profiles:
        _show_profiles()
        return

    # Validate required arguments for conversion
    if not args.input_tiff or not args.output_zarr:
        parser.error("Input and output paths are required for conversion")

    # Load additional config from file if specified
    extra_config = {}
    if args.config_file:
        try:
            with open(args.config_file, 'r') as f:
                extra_config = json.load(f)
        except Exception as e:
            console.print(f"[red]Error loading config file: {e}[/red]")
            sys.exit(1)

    # Convert arguments to function parameters
    success = convert_bigtiff_to_omezarr(
        input_tiff=args.input_tiff,
        output_zarr_dir=args.output_zarr,
        dimension_order=args.dimension_order,
        dtype=args.dtype,
        compressor=args.compressor,
        time_scale=args.time_scale,
        channel_scale=args.channel_scale,
        z_scale=args.z_scale,
        y_scale=args.y_scale,
        x_scale=args.x_scale,
        time_chunk=args.time_chunk,
        channel_chunk=args.channel_chunk,
        z_chunk=args.z_chunk,
        y_chunk=args.y_chunk,
        x_chunk=args.x_chunk,
        min_dimension_size=args.min_dimension_size,
        n_layers=args.n_layers,
        auto_chunk=args.auto_chunk,
        overwrite=args.overwrite,
        save_omexml=args.save_omexml,
        zarr_format=args.zarr_format,
        # Additional performance options
        workers=args.workers,
        chunk_memory_mb=args.chunk_memory_mb,
        use_numa=args.use_numa,
        use_async_io=args.use_async_io,
        profile=args.profile,
        compression_level=args.compression_level,
        verbose=args.verbose,
        **extra_config
    )

    sys.exit(0 if success else 1)


async def _show_system_info():
    """Display system information."""
    console.print(Panel("[bold blue]System Information[/bold blue]"))

    detector = HardwareDetector()
    await detector.detect_hardware()

    # CPU Information
    cpu_info = detector.hardware_info.get("cpu", {})
    table = Table(title="CPU Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Physical Cores", str(cpu_info.get("cores", "Unknown")))
    table.add_row("Logical Cores", str(cpu_info.get("threads", "Unknown")))
    table.add_row("Architecture", cpu_info.get("architecture", "Unknown"))
    table.add_row("Model", cpu_info.get("model", "Unknown"))
    table.add_row("Recommended Workers", str(cpu_info.get("recommended_workers", "Unknown")))

    console.print(table)

    # Memory Information
    memory_info = detector.hardware_info.get("memory", {})
    table = Table(title="Memory Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Memory", f"{memory_info.get('total_gb', 0):.1f} GB")
    table.add_row("Available Memory", f"{memory_info.get('available_gb', 0):.1f} GB")
    table.add_row("Memory Type", memory_info.get("type", "Unknown"))

    console.print(table)

    # Storage Information
    storage_info = detector.hardware_info.get("storage", {})
    table = Table(title="Storage Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Storage Type", storage_info.get("type", "Unknown"))
    table.add_row("Estimated Speed", storage_info.get("estimated_speed", "Unknown"))

    console.print(table)

    # Recommendations
    recommended_profile = detector.recommend_profile()
    console.print(f"\n[bold green]Recommended Profile: {recommended_profile.upper()}[/bold green]")


async def _run_benchmark():
    """Run performance benchmark."""
    console.print(Panel("[bold blue]Performance Benchmark[/bold blue]"))

    from core.benchmark import BenchmarkSuite

    benchmark = BenchmarkSuite()
    results = await benchmark.run_full_benchmark()

    # Display results
    table = Table(title="Benchmark Results")
    table.add_column("Test", style="cyan")
    table.add_column("Result", style="green")
    table.add_column("Unit", style="yellow")

    for test_name, result in results.items():
        if isinstance(result, dict) and "value" in result:
            table.add_row(test_name, f"{result['value']:.2f}", result.get("unit", ""))
        else:
            table.add_row(test_name, str(result), "")

    console.print(table)


def _show_profiles():
    """Show available optimization profiles."""
    console.print(Panel("[bold blue]Available Optimization Profiles[/bold blue]"))

    profile_manager = ConfigProfileManager()
    profiles = profile_manager.get_all_profiles()

    for name, profile in profiles.items():
        table = Table(title=f"{name.upper()} Profile")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Description", profile.get("description", ""))
        table.add_row("Workers", str(profile.get("workers", "auto")))
        table.add_row("Chunk Memory", f"{profile.get('chunk_memory_mb', 'auto')} MB")
        table.add_row("Compression", profile.get("compression", "blosc2-lz4"))
        table.add_row("Use NUMA", str(profile.get("use_numa", False)))
        table.add_row("Use Async I/O", str(profile.get("use_async_io", True)))

        console.print(table)
        console.print()


if __name__ == "__main__":
    main()