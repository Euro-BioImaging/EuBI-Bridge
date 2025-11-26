#!/usr/bin/env python3
"""
High-Performance BigTIFF to OME-NGFF Conversion Tool
Optimized for terabyte-scale data processing with parallel processing,
memory mapping, and hardware-specific optimizations.
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

from core.converter import HighPerformanceConverter
from core.progress_monitor import ProgressMonitor
from utils.hardware_detection import HardwareDetector
from utils.config_profiles import ConfigProfileManager
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()

def create_parser():
    """Create argument parser with all optimization options."""
    parser = argparse.ArgumentParser(
        description="High-Performance BigTIFF to OME-NGFF Converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python main.py input.tif output.zarr

  # High-performance conversion with all optimizations
  python main.py input.tif output.zarr --profile hpc --workers auto --use-numa

  # Custom optimization settings
  python main.py input.tif output.zarr --workers 16 --chunk-memory 2048 --compression blosc2-lz4

  # Web interface mode
  python main.py --web-server
        """
    )
    
    # Basic arguments
    parser.add_argument("input", nargs="?", help="Input BigTIFF file path")
    parser.add_argument("output", nargs="?", help="Output OME-NGFF directory path")
    
    # Performance profiles
    parser.add_argument("--profile", choices=["auto", "hpc", "workstation", "cloud", "memory-constrained"],
                       default="auto", help="Hardware optimization profile")
    
    # Parallel processing
    parser.add_argument("--workers", default="auto", 
                       help="Number of worker processes (auto, or integer)")
    parser.add_argument("--threads-per-worker", type=int, default=None,
                       help="Threads per worker process")
    parser.add_argument("--use-numa", action="store_true",
                       help="Enable NUMA-aware processing")
    
    # Memory management
    parser.add_argument("--chunk-memory", type=int, default=None,
                       help="Memory per chunk in MB")
    parser.add_argument("--max-memory", type=int, default=None,
                       help="Maximum total memory usage in MB")
    parser.add_argument("--use-mmap", action="store_true", default=True,
                       help="Use memory mapping for large files")
    
    # Compression
    parser.add_argument("--compression", choices=["blosc2-lz4", "blosc2-zstd", "lz4", "none"],
                       default="blosc2-lz4", help="Compression algorithm")
    parser.add_argument("--compression-level", type=int, default=3,
                       help="Compression level (1-9)")
    
    # I/O optimization
    parser.add_argument("--async-io", action="store_true", default=True,
                       help="Use asynchronous I/O")
    parser.add_argument("--io-buffer-size", type=int, default=None,
                       help="I/O buffer size in MB")
    
    # Pyramid settings
    parser.add_argument("--pyramid-levels", type=int, default=None,
                       help="Number of pyramid levels")
    parser.add_argument("--downsample-factor", type=int, default=2,
                       help="Downsampling factor between levels")
    
    # Monitoring and debugging
    parser.add_argument("--progress", action="store_true", default=True,
                       help="Show progress monitoring")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run performance benchmarks")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--profile-performance", action="store_true",
                       help="Enable performance profiling")
    
    # Web server mode
    parser.add_argument("--web-server", action="store_true",
                       help="Start web interface instead of CLI conversion")
    parser.add_argument("--port", type=int, default=5000,
                       help="Web server port")
    
    # Resumable conversion
    parser.add_argument("--resume", action="store_true",
                       help="Resume interrupted conversion")
    parser.add_argument("--checkpoint-interval", type=int, default=60,
                       help="Checkpoint interval in seconds")
    
    return parser

async def show_system_info():
    """Display system information and optimization recommendations."""
    detector = HardwareDetector()
    await detector.detect_hardware()
    
    console.print("\n[bold cyan]System Information[/bold cyan]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Details", style="white")
    table.add_column("Optimization", style="green")
    
    # CPU info
    cpu_info = detector.get_cpu_info()
    table.add_row(
        "CPU",
        f"{cpu_info['cores']} cores, {cpu_info['threads']} threads\n{cpu_info['model']}",
        f"Workers: {cpu_info['recommended_workers']}"
    )
    
    # Memory info
    mem_info = detector.get_memory_info()
    table.add_row(
        "Memory",
        f"{mem_info['total_gb']:.1f} GB total\n{mem_info['available_gb']:.1f} GB available",
        f"Chunk size: {mem_info['recommended_chunk_mb']} MB"
    )
    
    # Storage info
    storage_info = detector.get_storage_info()
    table.add_row(
        "Storage",
        f"Type: {storage_info['type']}\nSpeed: {storage_info['estimated_speed']}",
        f"Buffer: {storage_info['recommended_buffer_mb']} MB"
    )
    
    # NUMA info
    numa_info = detector.get_numa_info()
    if numa_info['available']:
        table.add_row(
            "NUMA",
            f"Nodes: {numa_info['nodes']}\nAffinity: Available",
            "Enable --use-numa for large datasets"
        )
    
    console.print(table)
    console.print()

async def run_conversion(args):
    """Run the conversion process with optimization."""
    if not args.input or not args.output:
        console.print("[red]Error: Input and output paths are required for conversion[/red]")
        return False
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[red]Error: Input file not found: {input_path}[/red]")
        return False
    
    # Show system info if verbose
    if args.verbose:
        await show_system_info()
    
    # Load configuration profile
    config_manager = ConfigProfileManager()
    config = await config_manager.get_profile_config(args.profile)
    
    # Override config with command line arguments
    if args.workers != "auto":
        config["workers"] = int(args.workers)
    if args.chunk_memory:
        config["chunk_memory_mb"] = args.chunk_memory
    if args.compression:
        config["compression"] = args.compression
    
    # Initialize converter
    converter = HighPerformanceConverter(config)
    
    # Initialize progress monitor
    progress_monitor = None
    if args.progress:
        progress_monitor = ProgressMonitor()
        await progress_monitor.start()
    
    try:
        # Run conversion
        console.print(f"\n[bold green]Starting conversion:[/bold green]")
        console.print(f"  Input:  {input_path}")
        console.print(f"  Output: {args.output}")
        console.print(f"  Profile: {args.profile}")
        console.print()
        
        success = await converter.convert(
            input_path=str(input_path),
            output_path=args.output,
            progress_monitor=progress_monitor,
            resume=args.resume
        )
        
        if success:
            console.print("\n[bold green]✓ Conversion completed successfully![/bold green]")
            
            # Show performance summary
            stats = await converter.get_performance_stats()
            console.print(f"\nPerformance Summary:")
            console.print(f"  Duration: {stats['duration']:.2f} seconds")
            console.print(f"  Throughput: {stats['throughput_mb_s']:.2f} MB/s")
            console.print(f"  Compression ratio: {stats['compression_ratio']:.2f}")
            
            return True
        else:
            console.print("\n[red]✗ Conversion failed[/red]")
            return False
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Conversion interrupted by user[/yellow]")
        if progress_monitor:
            await progress_monitor.stop()
        return False
    except Exception as e:
        console.print(f"\n[red]Error during conversion: {e}[/red]")
        if args.verbose:
            import traceback
            console.print(traceback.format_exc())
        return False
    finally:
        if progress_monitor:
            await progress_monitor.stop()

async def run_benchmark(args):
    """Run performance benchmarks."""
    from core.benchmark import BenchmarkSuite
    
    console.print("\n[bold cyan]Running Performance Benchmarks[/bold cyan]\n")
    
    benchmark = BenchmarkSuite()
    results = await benchmark.run_full_suite()
    
    # Display results
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Result", style="white")
    table.add_column("Score", style="green")
    
    for test_name, result in results.items():
        table.add_row(
            test_name,
            result['description'],
            f"{result['score']:.2f} {result['unit']}"
        )
    
    console.print(table)
    console.print()

async def start_web_server(port: int):
    """Start the web interface."""
    from app import create_app
    
    console.print(f"\n[bold green]Starting web interface on port {port}[/bold green]")
    console.print(f"Open your browser to: [link]http://localhost:{port}[/link]\n")
    
    app = create_app()
    
    # Run with uvicorn in production mode
    import uvicorn
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

async def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle web server mode
    if args.web_server:
        await start_web_server(args.port)
        return
    
    # Handle benchmark mode
    if args.benchmark:
        await run_benchmark(args)
        return
    
    # Show system info and exit if no conversion requested
    if not args.input and not args.output:
        await show_system_info()
        console.print("\n[yellow]Use --help for conversion options or --web-server for GUI[/yellow]")
        return
    
    # Run conversion
    success = await run_conversion(args)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    # Set up asyncio for Windows compatibility
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
