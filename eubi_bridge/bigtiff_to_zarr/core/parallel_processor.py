"""
Parallel processing engine for high-performance data conversion.
Implements worker pools, NUMA-aware processing, and load balancing.
"""

import asyncio
import multiprocessing as mp
import concurrent.futures
import threading
import queue
import time
import psutil
from typing import List, Dict, Any, Callable, Optional, Tuple
import numpy as np

class WorkerPool:
    """High-performance worker pool with NUMA awareness and load balancing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_workers = config.get("workers", psutil.cpu_count())
        self.threads_per_worker = config.get("threads_per_worker", 2)
        self.use_numa = config.get("use_numa", False)
        
        self.process_pool = None
        self.thread_pool = None
        self.numa_affinity = {}
        self.worker_stats = {}
        self.load_balancer = LoadBalancer()
    
    async def initialize(self):
        """Initialize worker pools with optimal configuration."""
        # Setup process pool for CPU-intensive tasks
        if self.use_numa:
            self.process_pool = NUMAProcessPool(
                max_workers=self.num_workers,
                numa_config=await self._detect_numa_topology()
            )
        else:
            self.process_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.num_workers
            )
        
        # Setup thread pool for I/O operations
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers * self.threads_per_worker
        )
        
        # Initialize worker monitoring
        await self._initialize_monitoring()
    
    async def _detect_numa_topology(self) -> Dict[str, Any]:
        """Detect NUMA topology for optimal worker placement."""
        numa_info = {
            "nodes": [],
            "cpu_mapping": {},
            "memory_mapping": {}
        }
        
        try:
            import numa
            if numa.available():
                num_nodes = numa.get_max_node() + 1
                for node in range(num_nodes):
                    node_cpus = numa.node_to_cpus(node)
                    numa_info["nodes"].append({
                        "node": node,
                        "cpus": node_cpus,
                        "memory_gb": numa.node_size(node) / (1024**3)
                    })
                    
                    for cpu in node_cpus:
                        numa_info["cpu_mapping"][cpu] = node
        except ImportError:
            # NUMA not available, use single node
            numa_info["nodes"] = [{
                "node": 0,
                "cpus": list(range(psutil.cpu_count())),
                "memory_gb": psutil.virtual_memory().total / (1024**3)
            }]
        
        return numa_info
    
    async def _initialize_monitoring(self):
        """Initialize worker performance monitoring."""
        for i in range(self.num_workers):
            self.worker_stats[i] = {
                "tasks_completed": 0,
                "total_time": 0.0,
                "current_load": 0.0,
                "memory_usage": 0.0,
                "cpu_usage": 0.0
            }
    
    async def submit_task(self, func: Callable, *args, **kwargs) -> asyncio.Future:
        """Submit task to optimal worker with load balancing."""
        # Choose optimal worker
        worker_id = await self.load_balancer.choose_worker(self.worker_stats)
        
        # Submit to appropriate pool
        if self._is_cpu_intensive_task(func):
            future = await self._submit_to_process_pool(func, worker_id, *args, **kwargs)
        else:
            future = await self._submit_to_thread_pool(func, worker_id, *args, **kwargs)
        
        # Update statistics
        asyncio.create_task(self._update_worker_stats(worker_id, future))
        
        return future
    
    def _is_cpu_intensive_task(self, func: Callable) -> bool:
        """Determine if task is CPU-intensive based on function attributes."""
        # Simple heuristic - could be enhanced with profiling data
        cpu_intensive_keywords = ["compress", "downsample", "transform", "calculate"]
        func_name = getattr(func, "__name__", str(func)).lower()
        return any(keyword in func_name for keyword in cpu_intensive_keywords)
    
    async def _submit_to_process_pool(self, func: Callable, worker_id: int, 
                                    *args, **kwargs) -> asyncio.Future:
        """Submit CPU-intensive task to process pool."""
        loop = asyncio.get_event_loop()
        
        # Wrap function with worker affinity if NUMA is enabled
        if self.use_numa and worker_id in self.numa_affinity:
            wrapped_func = self._wrap_with_affinity(func, self.numa_affinity[worker_id])
        else:
            wrapped_func = func
        
        return await loop.run_in_executor(self.process_pool, wrapped_func, *args, **kwargs)
    
    async def _submit_to_thread_pool(self, func: Callable, worker_id: int,
                                   *args, **kwargs) -> asyncio.Future:
        """Submit I/O task to thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, func, *args, **kwargs)
    
    def _wrap_with_affinity(self, func: Callable, cpu_affinity: List[int]) -> Callable:
        """Wrap function with CPU affinity setting."""
        def wrapped(*args, **kwargs):
            try:
                psutil.Process().cpu_affinity(cpu_affinity)
            except (AttributeError, OSError):
                pass  # CPU affinity not supported
            return func(*args, **kwargs)
        return wrapped
    
    async def _update_worker_stats(self, worker_id: int, future: asyncio.Future):
        """Update worker statistics after task completion."""
        start_time = time.time()
        try:
            await future
            end_time = time.time()
            task_time = end_time - start_time
            
            stats = self.worker_stats[worker_id]
            stats["tasks_completed"] += 1
            stats["total_time"] += task_time
            stats["current_load"] = max(0, stats["current_load"] - 1)
            
        except Exception:
            # Task failed, still update stats
            stats = self.worker_stats[worker_id]
            stats["current_load"] = max(0, stats["current_load"] - 1)
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        total_tasks = sum(stats["tasks_completed"] for stats in self.worker_stats.values())
        total_time = sum(stats["total_time"] for stats in self.worker_stats.values())
        avg_load = sum(stats["current_load"] for stats in self.worker_stats.values()) / len(self.worker_stats)
        
        return {
            "total_tasks": total_tasks,
            "total_time": total_time,
            "average_load": avg_load,
            "worker_stats": self.worker_stats.copy(),
            "throughput_tasks_per_sec": total_tasks / max(total_time, 0.001)
        }
    
    async def shutdown(self):
        """Shutdown worker pools gracefully."""
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)


class LoadBalancer:
    """Intelligent load balancer for worker selection."""
    
    def __init__(self):
        self.task_history = {}
        self.worker_preferences = {}
    
    async def choose_worker(self, worker_stats: Dict[int, Dict[str, Any]]) -> int:
        """Choose optimal worker based on current load and performance."""
        if not worker_stats:
            return 0
        
        # Calculate worker scores based on multiple factors
        scores = {}
        for worker_id, stats in worker_stats.items():
            # Lower is better
            load_score = stats["current_load"]
            cpu_score = stats.get("cpu_usage", 0) / 100.0
            memory_score = stats.get("memory_usage", 0) / 100.0
            
            # Performance score (higher is better, so invert)
            if stats["tasks_completed"] > 0:
                avg_time = stats["total_time"] / stats["tasks_completed"]
                perf_score = 1.0 / max(avg_time, 0.001)
            else:
                perf_score = 1.0
            
            # Combined score (lower is better)
            total_score = (load_score * 0.4 + 
                          cpu_score * 0.3 + 
                          memory_score * 0.2 - 
                          perf_score * 0.1)
            
            scores[worker_id] = total_score
        
        # Choose worker with lowest score
        best_worker = min(scores.keys(), key=lambda w: scores[w])
        
        # Update load
        worker_stats[best_worker]["current_load"] += 1
        
        return best_worker


class NUMAProcessPool:
    """NUMA-aware process pool for optimal memory locality."""
    
    def __init__(self, max_workers: int, numa_config: Dict[str, Any]):
        self.max_workers = max_workers
        self.numa_config = numa_config
        self.processes = {}
        self.task_queue = queue.Queue()
        self.result_futures = {}
        self.shutdown_event = threading.Event()
    
    def submit(self, func: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit task with NUMA-aware scheduling."""
        future = concurrent.futures.Future()
        task_id = id(future)
        
        # Choose optimal NUMA node
        numa_node = self._choose_numa_node()
        
        # Submit task
        task = {
            "id": task_id,
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "numa_node": numa_node,
            "future": future
        }
        
        self.task_queue.put(task)
        return future
    
    def _choose_numa_node(self) -> int:
        """Choose optimal NUMA node based on current load."""
        # Simple round-robin for now
        nodes = [node["node"] for node in self.numa_config["nodes"]]
        return nodes[len(self.result_futures) % len(nodes)] if nodes else 0
    
    def shutdown(self, wait: bool = True):
        """Shutdown process pool."""
        self.shutdown_event.set()
        if wait:
            # Wait for all processes to finish
            for process in self.processes.values():
                process.join(timeout=5.0)


class ParallelChunkProcessor:
    """Specialized processor for parallel chunk operations."""
    
    def __init__(self, worker_pool: WorkerPool):
        self.worker_pool = worker_pool
        self.chunk_cache = {}
        self.prefetch_queue = asyncio.Queue(maxsize=100)
    
    async def process_chunks_parallel(self, chunks: List[Dict[str, Any]], 
                                    process_func: Callable,
                                    max_concurrent: int = None) -> List[Any]:
        """Process chunks in parallel with optimal scheduling."""
        if max_concurrent is None:
            max_concurrent = self.worker_pool.num_workers * 2
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_chunk(chunk_info):
            async with semaphore:
                return await self.worker_pool.submit_task(process_func, chunk_info)
        
        # Start prefetching
        prefetch_task = asyncio.create_task(self._prefetch_chunks(chunks))
        
        try:
            # Process all chunks
            tasks = [process_single_chunk(chunk) for chunk in chunks]
            results = await asyncio.gather(*tasks)
            return results
        finally:
            prefetch_task.cancel()
    
    async def _prefetch_chunks(self, chunks: List[Dict[str, Any]]):
        """Prefetch chunk data to improve I/O efficiency."""
        for chunk in chunks:
            # Simple prefetch strategy - could be enhanced
            await asyncio.sleep(0.001)  # Yield control
            # Actual prefetching would happen here
    
    async def get_chunk_stats(self) -> Dict[str, Any]:
        """Get chunk processing statistics."""
        return {
            "cache_hits": len(self.chunk_cache),
            "prefetch_queue_size": self.prefetch_queue.qsize(),
            "cache_memory_mb": sum(
                chunk.nbytes if hasattr(chunk, 'nbytes') else 0 
                for chunk in self.chunk_cache.values()
            ) / (1024 * 1024)
        }


# Utility functions for parallel operations
async def parallel_map(func: Callable, items: List[Any], 
                      worker_pool: WorkerPool, 
                      max_concurrent: int = None) -> List[Any]:
    """Parallel map operation with worker pool."""
    if max_concurrent is None:
        max_concurrent = worker_pool.num_workers
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_item(item):
        async with semaphore:
            return await worker_pool.submit_task(func, item)
    
    tasks = [process_item(item) for item in items]
    return await asyncio.gather(*tasks)


def cpu_intensive_task(func: Callable) -> Callable:
    """Decorator to mark function as CPU-intensive."""
    func._cpu_intensive = True
    return func


def io_intensive_task(func: Callable) -> Callable:
    """Decorator to mark function as I/O-intensive."""
    func._io_intensive = True
    return func
