"""
Comprehensive benchmarking suite for performance optimization.
Tests various hardware configurations and optimization strategies.
"""

import asyncio
import time
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import psutil

from .compression_engine import CompressionEngine, benchmark_compressors
from .memory_manager import MemoryManager
from .async_io import AsyncIOManager
from .system_optimizer import SystemOptimizer


class BenchmarkSuite:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.results = {}
        
        # Default configurations for testing
        self.test_configs = {
            "baseline": {
                "workers": 1,
                "compression": "none",
                "use_mmap": False,
                "use_numa": False,
                "chunk_memory_mb": 64
            },
            "optimized": {
                "workers": psutil.cpu_count(),
                "compression": "blosc2-lz4",
                "use_mmap": True,
                "use_numa": False,
                "chunk_memory_mb": 256
            },
            "high_performance": {
                "workers": psutil.cpu_count() * 2,
                "compression": "blosc2-lz4",
                "use_mmap": True,
                "use_numa": True,
                "chunk_memory_mb": 512
            }
        }
    
    async def run_full_suite(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        print("Starting comprehensive benchmark suite...")
        
        suite_start = time.time()
        
        # System information
        self.results["system_info"] = await self._benchmark_system_info()
        
        # CPU and memory benchmarks
        self.results["cpu_benchmark"] = await self._benchmark_cpu_performance()
        self.results["memory_benchmark"] = await self._benchmark_memory_performance()
        
        # I/O benchmarks
        self.results["io_benchmark"] = await self._benchmark_io_performance()
        
        # Compression benchmarks
        self.results["compression_benchmark"] = await self._benchmark_compression()
        
        # Parallel processing benchmarks
        self.results["parallel_benchmark"] = await self._benchmark_parallel_processing()
        
        # Integration benchmarks (simulated conversion)
        self.results["integration_benchmark"] = await self._benchmark_integration()
        
        # Configuration comparison
        self.results["config_comparison"] = await self._benchmark_configurations()
        
        suite_duration = time.time() - suite_start
        self.results["suite_summary"] = {
            "total_duration": suite_duration,
            "timestamp": time.time(),
            "overall_score": self._calculate_overall_score()
        }
        
        return self.results
    
    async def _benchmark_system_info(self) -> Dict[str, Any]:
        """Benchmark system hardware capabilities."""
        info = {
            "cpu": {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "max_frequency": None,
                "features": []
            },
            "memory": {
                "total_gb": psutil.virtual_memory().total / (1024**3),
                "available_gb": psutil.virtual_memory().available / (1024**3)
            },
            "storage": {},
            "numa": {"available": False}
        }
        
        # CPU frequency
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                info["cpu"]["max_frequency"] = cpu_freq.max
        except:
            pass
        
        # Storage info
        try:
            disk_usage = psutil.disk_usage(self.temp_dir)
            info["storage"] = {
                "free_gb": disk_usage.free / (1024**3),
                "total_gb": disk_usage.total / (1024**3)
            }
        except:
            pass
        
        # NUMA availability
        try:
            import numa
            if numa.available():
                info["numa"] = {
                    "available": True,
                    "nodes": numa.get_max_node() + 1
                }
        except ImportError:
            pass
        
        return info
    
    async def _benchmark_cpu_performance(self) -> Dict[str, Any]:
        """Benchmark CPU performance with various workloads."""
        results = {}
        
        # CPU-intensive computation benchmark
        print("Running CPU benchmark...")
        
        def cpu_intensive_task(size: int = 1000000):
            """CPU-intensive task for benchmarking."""
            start = time.time()
            
            # Matrix multiplication
            a = np.random.rand(size // 1000, size // 1000).astype(np.float32)
            b = np.random.rand(size // 1000, size // 1000).astype(np.float32)
            c = np.dot(a, b)
            
            return time.time() - start
        
        # Single-threaded performance
        single_time = await asyncio.get_event_loop().run_in_executor(None, cpu_intensive_task)
        results["single_thread_time"] = single_time
        results["single_thread_score"] = 1000.0 / single_time  # Higher is better
        
        # Multi-threaded performance
        from concurrent.futures import ThreadPoolExecutor
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=psutil.cpu_count()) as executor:
            tasks = [
                executor.submit(cpu_intensive_task, 1000000 // psutil.cpu_count()) 
                for _ in range(psutil.cpu_count())
            ]
            await asyncio.gather(*[
                asyncio.get_event_loop().run_in_executor(None, task.result)
                for task in tasks
            ])
        
        multi_time = time.time() - start_time
        results["multi_thread_time"] = multi_time
        results["multi_thread_score"] = 1000.0 / multi_time
        results["parallel_efficiency"] = (single_time / multi_time) / psutil.cpu_count()
        
        return results
    
    async def _benchmark_memory_performance(self) -> Dict[str, Any]:
        """Benchmark memory allocation and access patterns."""
        results = {}
        print("Running memory benchmark...")
        
        # Memory allocation benchmark
        sizes = [1024, 10240, 102400, 1024000]  # 1KB to 1GB (in KB)
        
        for size_kb in sizes:
            size_bytes = size_kb * 1024
            
            # Allocation time
            start = time.time()
            data = np.empty(size_bytes // 8, dtype=np.float64)
            alloc_time = time.time() - start
            
            # Sequential access
            start = time.time()
            data.fill(1.0)
            seq_time = time.time() - start
            
            # Random access
            start = time.time()
            indices = np.random.randint(0, len(data), size=min(10000, len(data)))
            data[indices] = 2.0
            random_time = time.time() - start
            
            results[f"size_{size_kb}kb"] = {
                "allocation_time": alloc_time,
                "sequential_access_time": seq_time,
                "random_access_time": random_time,
                "sequential_bandwidth_mb_s": (size_bytes / (1024*1024)) / max(seq_time, 0.001),
                "random_bandwidth_mb_s": (len(indices) * 8 / (1024*1024)) / max(random_time, 0.001)
            }
            
            del data  # Free memory
        
        return results
    
    async def _benchmark_io_performance(self) -> Dict[str, Any]:
        """Benchmark I/O performance for different access patterns."""
        results = {}
        print("Running I/O benchmark...")
        
        test_file = Path(self.temp_dir) / "benchmark_io_test.dat"
        
        try:
            # Test different file sizes
            sizes_mb = [1, 10, 100, 1000]  # 1MB to 1GB
            
            for size_mb in sizes_mb:
                if psutil.disk_usage(self.temp_dir).free < size_mb * 1024 * 1024 * 2:
                    continue  # Skip if not enough space
                
                size_bytes = size_mb * 1024 * 1024
                data = np.random.bytes(size_bytes)
                
                # Sequential write
                start = time.time()
                with open(test_file, 'wb') as f:
                    f.write(data)
                write_time = time.time() - start
                
                # Sequential read
                start = time.time()
                with open(test_file, 'rb') as f:
                    read_data = f.read()
                read_time = time.time() - start
                
                # Async I/O test
                config = {"io_buffer_size_mb": 64, "max_concurrent_reads": 4}
                async_io = AsyncIOManager(config)
                
                start = time.time()
                async_data = await async_io.read_chunk_async(str(test_file), 0, size_bytes)
                async_read_time = time.time() - start
                
                results[f"size_{size_mb}mb"] = {
                    "sequential_write_mb_s": size_mb / max(write_time, 0.001),
                    "sequential_read_mb_s": size_mb / max(read_time, 0.001),
                    "async_read_mb_s": size_mb / max(async_read_time, 0.001),
                    "write_time": write_time,
                    "read_time": read_time,
                    "async_read_time": async_read_time
                }
                
                test_file.unlink()  # Clean up
                
        except Exception as e:
            results["error"] = str(e)
        finally:
            if test_file.exists():
                test_file.unlink()
        
        return results
    
    async def _benchmark_compression(self) -> Dict[str, Any]:
        """Benchmark different compression algorithms."""
        results = {}
        print("Running compression benchmark...")
        
        # Generate test data with different characteristics
        test_datasets = {
            "random": np.random.randint(0, 256, size=(1024, 1024), dtype=np.uint8),
            "structured": np.tile(np.arange(256, dtype=np.uint8), (1024, 4)),
            "sparse": np.zeros((1024, 1024), dtype=np.uint8),
            "float": np.random.rand(512, 512).astype(np.float32)
        }
        
        # Add some structure to sparse data
        test_datasets["sparse"][::10, ::10] = 255
        
        compression_configs = [
            {"compression": "blosc2-lz4", "compression_level": 1},
            {"compression": "blosc2-lz4", "compression_level": 3},
            {"compression": "blosc2-zstd", "compression_level": 3},
            {"compression": "lz4", "compression_level": 1},
            {"compression": "none", "compression_level": 0}
        ]
        
        for data_name, data in test_datasets.items():
            results[data_name] = {}
            
            for config in compression_configs:
                try:
                    engine = CompressionEngine(config)
                    compressor = engine.get_compressor()
                    
                    if compressor is None and config["compression"] != "none":
                        continue
                    
                    # Compression benchmark
                    start = time.time()
                    if compressor:
                        compressed = engine.compress_array(data)
                    else:
                        compressed = data.tobytes()
                    compression_time = time.time() - start
                    
                    # Decompression benchmark
                    start = time.time()
                    if compressor:
                        decompressed = engine.decompress_array(compressed, data.shape, data.dtype)
                    else:
                        decompressed = np.frombuffer(compressed, dtype=data.dtype).reshape(data.shape)
                    decompression_time = time.time() - start
                    
                    # Calculate metrics
                    original_size = data.nbytes
                    compressed_size = len(compressed)
                    compression_ratio = original_size / compressed_size
                    
                    config_name = f"{config['compression']}_level{config['compression_level']}"
                    results[data_name][config_name] = {
                        "compression_ratio": compression_ratio,
                        "compression_time": compression_time,
                        "decompression_time": decompression_time,
                        "compression_speed_mb_s": (original_size / (1024*1024)) / max(compression_time, 0.001),
                        "decompression_speed_mb_s": (original_size / (1024*1024)) / max(decompression_time, 0.001),
                        "total_time": compression_time + decompression_time
                    }
                    
                except Exception as e:
                    config_name = f"{config['compression']}_level{config['compression_level']}"
                    results[data_name][config_name] = {"error": str(e)}
        
        return results
    
    async def _benchmark_parallel_processing(self) -> Dict[str, Any]:
        """Benchmark parallel processing scalability."""
        results = {}
        print("Running parallel processing benchmark...")
        
        def parallel_task(data_size: int = 1000000):
            """Simulated data processing task."""
            data = np.random.rand(data_size).astype(np.float32)
            # Simulate some computation
            result = np.sum(data * data) + np.mean(data)
            return result
        
        # Test with different worker counts
        worker_counts = [1, 2, 4, psutil.cpu_count(), psutil.cpu_count() * 2]
        task_count = 16
        
        for workers in worker_counts:
            if workers > psutil.cpu_count() * 4:  # Reasonable limit
                continue
            
            from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
            
            # Process pool benchmark
            start = time.time()
            with ProcessPoolExecutor(max_workers=workers) as executor:
                tasks = [executor.submit(parallel_task) for _ in range(task_count)]
                process_results = [task.result() for task in tasks]
            process_time = time.time() - start
            
            # Thread pool benchmark
            start = time.time()
            with ThreadPoolExecutor(max_workers=workers) as executor:
                tasks = [executor.submit(parallel_task) for _ in range(task_count)]
                thread_results = [task.result() for task in tasks]
            thread_time = time.time() - start
            
            results[f"workers_{workers}"] = {
                "process_pool_time": process_time,
                "thread_pool_time": thread_time,
                "process_throughput": task_count / process_time,
                "thread_throughput": task_count / thread_time,
                "process_efficiency": (1.0 / process_time) / workers if workers > 0 else 0,
                "thread_efficiency": (1.0 / thread_time) / workers if workers > 0 else 0
            }
        
        return results
    
    async def _benchmark_integration(self) -> Dict[str, Any]:
        """Benchmark integrated conversion simulation."""
        results = {}
        print("Running integration benchmark...")
        
        # Simulate conversion of different sized datasets
        test_sizes = [
            (512, 512, 1, 1, 1),      # 0.25 MB
            (1024, 1024, 1, 1, 1),    # 1 MB
            (2048, 2048, 1, 1, 1),    # 4 MB
            (1024, 1024, 10, 1, 1),   # 10 MB
            (2048, 2048, 10, 1, 1)    # 40 MB
        ]
        
        for i, shape in enumerate(test_sizes):
            if np.prod(shape) * 2 > 100_000_000:  # Limit to 100MB
                continue
            
            try:
                # Generate test data
                data = np.random.randint(0, 65535, size=shape, dtype=np.uint16)
                data_size_mb = data.nbytes / (1024 * 1024)
                
                # Simulate optimized processing pipeline
                start = time.time()
                
                # Memory management
                config = {"chunk_memory_mb": 64, "use_mmap": True}
                memory_manager = MemoryManager(config)
                
                # Compression
                compression_config = {"compression": "blosc2-lz4", "compression_level": 3}
                compression_engine = CompressionEngine(compression_config)
                
                # Simulate chunk processing
                chunk_size = (256, 256, 1, 1, 1)[:len(shape)]
                chunks_processed = 0
                
                # Process in chunks
                for z in range(0, shape[2] if len(shape) > 2 else 1, chunk_size[2]):
                    for y in range(0, shape[1] if len(shape) > 1 else 1, chunk_size[1]):
                        for x in range(0, shape[0], chunk_size[0]):
                            # Extract chunk
                            if len(shape) == 5:  # Full 5D
                                chunk = data[x:x+chunk_size[0], 
                                           y:y+chunk_size[1], 
                                           z:z+chunk_size[2], :, :]
                            elif len(shape) == 3:  # 3D
                                chunk = data[x:x+chunk_size[0], 
                                           y:y+chunk_size[1], 
                                           z:z+chunk_size[2]]
                            else:  # 2D
                                chunk = data[x:x+chunk_size[0], 
                                           y:y+chunk_size[1]]
                            
                            # Simulate compression
                            compressed = compression_engine.compress_array(chunk)
                            
                            chunks_processed += 1
                
                processing_time = time.time() - start
                
                results[f"test_case_{i}"] = {
                    "shape": shape,
                    "data_size_mb": data_size_mb,
                    "processing_time": processing_time,
                    "throughput_mb_s": data_size_mb / max(processing_time, 0.001),
                    "chunks_processed": chunks_processed
                }
                
            except Exception as e:
                results[f"test_case_{i}"] = {
                    "shape": shape,
                    "error": str(e)
                }
        
        return results
    
    async def _benchmark_configurations(self) -> Dict[str, Any]:
        """Compare different configuration profiles."""
        results = {}
        print("Running configuration comparison...")
        
        # Simple task for testing configurations
        def test_task(config_name: str, config: Dict[str, Any]):
            # Simulate configuration-dependent performance
            base_time = 1.0
            
            # Adjust based on configuration
            if config.get("compression") != "none":
                base_time *= 0.8  # Compression reduces I/O time
            
            if config.get("use_mmap"):
                base_time *= 0.9  # Memory mapping helps
            
            workers = config.get("workers", 1)
            if workers > 1:
                base_time /= min(workers, psutil.cpu_count())  # Parallel speedup
            
            # Simulate some work
            time.sleep(base_time / 10)  # Scale down for benchmark
            
            return base_time
        
        for config_name, config in self.test_configs.items():
            start = time.time()
            simulated_time = test_task(config_name, config)
            actual_time = time.time() - start
            
            results[config_name] = {
                "config": config,
                "simulated_performance": 1.0 / simulated_time,  # Higher is better
                "actual_time": actual_time,
                "estimated_speedup": 1.0 / simulated_time  # Relative to baseline
            }
        
        return results
    
    def _calculate_overall_score(self) -> float:
        """Calculate overall performance score."""
        scores = []
        
        # CPU score
        if "cpu_benchmark" in self.results:
            cpu_score = self.results["cpu_benchmark"].get("multi_thread_score", 0)
            scores.append(cpu_score)
        
        # I/O score (average throughput)
        if "io_benchmark" in self.results:
            io_scores = []
            for size_test in self.results["io_benchmark"].values():
                if isinstance(size_test, dict) and "sequential_read_mb_s" in size_test:
                    io_scores.append(size_test["sequential_read_mb_s"])
            
            if io_scores:
                scores.append(sum(io_scores) / len(io_scores))
        
        # Compression score
        if "compression_benchmark" in self.results:
            compression_scores = []
            for data_type in self.results["compression_benchmark"].values():
                if isinstance(data_type, dict):
                    for config_result in data_type.values():
                        if isinstance(config_result, dict) and "compression_speed_mb_s" in config_result:
                            compression_scores.append(config_result["compression_speed_mb_s"])
            
            if compression_scores:
                scores.append(sum(compression_scores) / len(compression_scores))
        
        # Integration score
        if "integration_benchmark" in self.results:
            integration_scores = []
            for test_result in self.results["integration_benchmark"].values():
                if isinstance(test_result, dict) and "throughput_mb_s" in test_result:
                    integration_scores.append(test_result["throughput_mb_s"])
            
            if integration_scores:
                scores.append(sum(integration_scores) / len(integration_scores))
        
        return sum(scores) / len(scores) if scores else 0.0
    
    async def run_quick_benchmark(self) -> Dict[str, Any]:
        """Run a quick benchmark for basic performance assessment."""
        print("Running quick benchmark...")
        
        results = {
            "system_info": await self._benchmark_system_info(),
            "cpu_quick": await self._quick_cpu_test(),
            "memory_quick": await self._quick_memory_test(),
            "io_quick": await self._quick_io_test()
        }
        
        return results
    
    async def _quick_cpu_test(self) -> Dict[str, Any]:
        """Quick CPU performance test."""
        def cpu_task():
            start = time.time()
            # Simple computation
            result = sum(i * i for i in range(100000))
            return time.time() - start
        
        cpu_time = await asyncio.get_event_loop().run_in_executor(None, cpu_task)
        
        return {
            "computation_time": cpu_time,
            "score": 1000.0 / cpu_time
        }
    
    async def _quick_memory_test(self) -> Dict[str, Any]:
        """Quick memory performance test."""
        size_mb = 10
        data = np.random.rand(size_mb * 1024 * 1024 // 8).astype(np.float64)
        
        start = time.time()
        data.fill(1.0)
        memory_time = time.time() - start
        
        return {
            "access_time": memory_time,
            "bandwidth_mb_s": size_mb / max(memory_time, 0.001)
        }
    
    async def _quick_io_test(self) -> Dict[str, Any]:
        """Quick I/O performance test."""
        test_file = Path(self.temp_dir) / "quick_io_test.dat"
        size_mb = 5
        data = np.random.bytes(size_mb * 1024 * 1024)
        
        try:
            # Write test
            start = time.time()
            with open(test_file, 'wb') as f:
                f.write(data)
            write_time = time.time() - start
            
            # Read test
            start = time.time()
            with open(test_file, 'rb') as f:
                read_data = f.read()
            read_time = time.time() - start
            
            return {
                "write_mb_s": size_mb / max(write_time, 0.001),
                "read_mb_s": size_mb / max(read_time, 0.001)
            }
        
        finally:
            if test_file.exists():
                test_file.unlink()
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate a human-readable benchmark report."""
        if not self.results:
            return "No benchmark results available."
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("BigTIFF to OME-NGFF Converter - Performance Benchmark Report")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # System information
        if "system_info" in self.results:
            sys_info = self.results["system_info"]
            report_lines.append("System Information:")
            report_lines.append(f"  CPU: {sys_info['cpu']['logical_cores']} cores")
            report_lines.append(f"  Memory: {sys_info['memory']['total_gb']:.1f} GB")
            report_lines.append(f"  Storage: {sys_info['storage'].get('total_gb', 'Unknown')} GB")
            report_lines.append("")
        
        # Performance summary
        if "suite_summary" in self.results:
            summary = self.results["suite_summary"]
            report_lines.append(f"Overall Performance Score: {summary['overall_score']:.2f}")
            report_lines.append(f"Benchmark Duration: {summary['total_duration']:.2f} seconds")
            report_lines.append("")
        
        # Configuration recommendations
        report_lines.append("Recommended Configuration:")
        if "config_comparison" in self.results:
            best_config = max(
                self.results["config_comparison"].items(),
                key=lambda x: x[1].get("estimated_speedup", 0)
            )
            report_lines.append(f"  Profile: {best_config[0]}")
            for key, value in best_config[1]["config"].items():
                report_lines.append(f"  {key}: {value}")
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
        
        return report


# Utility functions for standalone benchmarking
async def quick_system_benchmark() -> Dict[str, Any]:
    """Run a quick system benchmark for immediate feedback."""
    benchmark = BenchmarkSuite()
    return await benchmark.run_quick_benchmark()


async def comprehensive_benchmark(output_file: Optional[str] = None) -> Dict[str, Any]:
    """Run comprehensive benchmark and optionally save report."""
    benchmark = BenchmarkSuite()
    results = await benchmark.run_full_suite()
    
    if output_file:
        report = benchmark.generate_report(output_file)
        print(f"Benchmark report saved to: {output_file}")
    
    return results
