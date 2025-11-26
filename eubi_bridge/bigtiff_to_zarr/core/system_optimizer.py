"""
System-level optimizations for high-performance data processing.
Handles CPU affinity, NUMA awareness, and resource management.
"""

import os
import psutil
import asyncio
from typing import Dict, Any, List, Optional
import threading
import multiprocessing as mp

try:
    import numa
    NUMA_AVAILABLE = True
except ImportError:
    NUMA_AVAILABLE = False


class SystemOptimizer:
    """System-level performance optimizer."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.original_affinity = None
        self.original_priority = None
        self.numa_topology = None
        self.optimization_applied = False
        
        # Resource limits
        self.max_memory_mb = config.get("max_memory_mb", None)
        self.cpu_affinity = config.get("cpu_affinity", None)
        self.use_numa = config.get("use_numa", False)
        self.high_priority = config.get("high_priority", False)
    
    async def optimize_system(self):
        """Apply system-level optimizations."""
        try:
            # Store original settings
            await self._store_original_settings()
            
            # Apply CPU optimizations
            await self._optimize_cpu_usage()
            
            # Apply memory optimizations
            await self._optimize_memory_usage()
            
            # Apply I/O optimizations
            await self._optimize_io_settings()
            
            # Apply NUMA optimizations
            if self.use_numa and NUMA_AVAILABLE:
                await self._optimize_numa_settings()
            
            self.optimization_applied = True
            
        except Exception as e:
            print(f"Warning: Failed to apply some system optimizations: {e}")
    
    async def _store_original_settings(self):
        """Store original system settings for restoration."""
        try:
            process = psutil.Process()
            self.original_affinity = process.cpu_affinity()
            self.original_priority = process.nice()
        except (AttributeError, OSError):
            pass
    
    async def _optimize_cpu_usage(self):
        """Optimize CPU usage and affinity."""
        try:
            process = psutil.Process()
            
            # Set CPU affinity if specified
            if self.cpu_affinity:
                process.cpu_affinity(self.cpu_affinity)
            elif self.use_numa and NUMA_AVAILABLE:
                # Let NUMA optimization handle affinity
                pass
            else:
                # Use all available CPUs
                all_cpus = list(range(psutil.cpu_count()))
                process.cpu_affinity(all_cpus)
            
            # Set process priority
            if self.high_priority:
                try:
                    if os.name == 'nt':  # Windows
                        process.nice(psutil.HIGH_PRIORITY_CLASS)
                    else:  # Unix-like
                        process.nice(-10)  # Higher priority
                except (AttributeError, OSError, PermissionError):
                    pass  # Might require elevated privileges
            
        except Exception as e:
            print(f"Warning: CPU optimization failed: {e}")
    
    async def _optimize_memory_usage(self):
        """Optimize memory usage settings."""
        try:
            # Set memory limits if specified
            if self.max_memory_mb:
                import resource
                max_memory_bytes = self.max_memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
            
            # Optimize memory allocation
            self._optimize_malloc_settings()
            
        except Exception as e:
            print(f"Warning: Memory optimization failed: {e}")
    
    def _optimize_malloc_settings(self):
        """Optimize memory allocation settings."""
        try:
            # Set environment variables for better memory allocation
            # These affect memory allocators like glibc malloc
            
            # Reduce memory fragmentation
            os.environ['MALLOC_MMAP_THRESHOLD_'] = '65536'  # 64KB
            os.environ['MALLOC_TRIM_THRESHOLD_'] = '131072'  # 128KB
            
            # For jemalloc (if available)
            os.environ['MALLOC_CONF'] = 'dirty_decay_ms:1000,muzzy_decay_ms:1000'
            
        except Exception:
            pass
    
    async def _optimize_io_settings(self):
        """Optimize I/O settings for performance."""
        try:
            # Set process I/O priority (Linux only)
            if hasattr(psutil.Process(), 'ionice'):
                process = psutil.Process()
                try:
                    # Set to real-time I/O priority class
                    process.ionice(psutil.IOPRIO_CLASS_RT, value=4)
                except (OSError, AttributeError):
                    # Fallback to best-effort high priority
                    try:
                        process.ionice(psutil.IOPRIO_CLASS_BE, value=1)
                    except (OSError, AttributeError):
                        pass
            
            # Optimize file system cache behavior
            self._optimize_filesystem_cache()
            
        except Exception as e:
            print(f"Warning: I/O optimization failed: {e}")
    
    def _optimize_filesystem_cache(self):
        """Optimize filesystem cache settings."""
        try:
            # Linux-specific optimizations
            if os.name == 'posix':
                # Increase readahead for sequential I/O
                # This would require system-level configuration
                pass
        except Exception:
            pass
    
    async def _optimize_numa_settings(self):
        """Optimize NUMA settings for memory locality."""
        if not NUMA_AVAILABLE:
            return
        
        try:
            # Detect NUMA topology
            self.numa_topology = await self._detect_numa_topology()
            
            # Set NUMA policy for better memory locality
            self._set_numa_policy()
            
        except Exception as e:
            print(f"Warning: NUMA optimization failed: {e}")
    
    async def _detect_numa_topology(self) -> Dict[str, Any]:
        """Detect NUMA topology."""
        topology = {
            "nodes": [],
            "total_nodes": 0,
            "current_node": 0
        }
        
        try:
            if numa.available():
                topology["total_nodes"] = numa.get_max_node() + 1
                
                for node in range(topology["total_nodes"]):
                    node_info = {
                        "node": node,
                        "cpus": numa.node_to_cpus(node),
                        "memory_gb": numa.node_size(node) / (1024**3),
                        "free_memory_gb": numa.node_size(node) / (1024**3)  # Simplified
                    }
                    topology["nodes"].append(node_info)
                
                # Get current node
                topology["current_node"] = numa.get_mempolicy()[1]
        
        except Exception:
            pass
        
        return topology
    
    def _set_numa_policy(self):
        """Set optimal NUMA memory policy."""
        try:
            if not self.numa_topology or self.numa_topology["total_nodes"] <= 1:
                return
            
            # Set memory policy to local allocation
            numa.set_mempolicy(numa.MPOL_LOCAL)
            
            # Bind to current NUMA node for better locality
            current_node = self.numa_topology["current_node"]
            numa.set_membind([current_node])
            
        except Exception:
            pass
    
    async def setup_numa_affinity(self, num_workers: int):
        """Setup NUMA affinity for worker processes."""
        if not NUMA_AVAILABLE or not self.numa_topology:
            return
        
        try:
            total_nodes = self.numa_topology["total_nodes"]
            if total_nodes <= 1:
                return
            
            # Distribute workers across NUMA nodes
            workers_per_node = max(1, num_workers // total_nodes)
            
            affinity_map = {}
            worker_id = 0
            
            for node_info in self.numa_topology["nodes"]:
                node_cpus = node_info["cpus"]
                
                for _ in range(min(workers_per_node, len(node_cpus))):
                    if worker_id < num_workers:
                        affinity_map[worker_id] = {
                            "numa_node": node_info["node"],
                            "cpu_list": node_cpus,
                            "preferred_cpus": node_cpus[:workers_per_node]
                        }
                        worker_id += 1
            
            return affinity_map
            
        except Exception as e:
            print(f"Warning: NUMA affinity setup failed: {e}")
            return {}
    
    def set_worker_affinity(self, worker_id: int, affinity_info: Dict[str, Any]):
        """Set CPU affinity for a specific worker."""
        try:
            if "preferred_cpus" in affinity_info:
                psutil.Process().cpu_affinity(affinity_info["preferred_cpus"])
            
            if NUMA_AVAILABLE and "numa_node" in affinity_info:
                numa.set_membind([affinity_info["numa_node"]])
                
        except Exception:
            pass
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        info = {
            "cpu": self._get_cpu_info(),
            "memory": self._get_memory_info(),
            "storage": self._get_storage_info(),
            "numa": self._get_numa_info(),
            "optimization_status": self.optimization_applied
        }
        
        return info
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information."""
        try:
            cpu_freq = psutil.cpu_freq()
            return {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "current_freq_mhz": cpu_freq.current if cpu_freq else None,
                "max_freq_mhz": cpu_freq.max if cpu_freq else None,
                "usage_percent": psutil.cpu_percent(interval=1),
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
            }
        except Exception:
            return {"error": "Could not retrieve CPU info"}
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory information."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3),
                "percent_used": memory.percent,
                "swap_total_gb": swap.total / (1024**3),
                "swap_used_gb": swap.used / (1024**3),
                "swap_percent": swap.percent
            }
        except Exception:
            return {"error": "Could not retrieve memory info"}
    
    def _get_storage_info(self) -> Dict[str, Any]:
        """Get storage information."""
        try:
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            return {
                "total_gb": disk_usage.total / (1024**3),
                "free_gb": disk_usage.free / (1024**3),
                "used_gb": disk_usage.used / (1024**3),
                "percent_used": (disk_usage.used / disk_usage.total) * 100,
                "read_count": disk_io.read_count if disk_io else None,
                "write_count": disk_io.write_count if disk_io else None,
                "read_bytes": disk_io.read_bytes if disk_io else None,
                "write_bytes": disk_io.write_bytes if disk_io else None
            }
        except Exception:
            return {"error": "Could not retrieve storage info"}
    
    def _get_numa_info(self) -> Dict[str, Any]:
        """Get NUMA information."""
        if not NUMA_AVAILABLE:
            return {"available": False, "reason": "NUMA library not available"}
        
        try:
            if not numa.available():
                return {"available": False, "reason": "NUMA not supported by system"}
            
            return {
                "available": True,
                "total_nodes": numa.get_max_node() + 1,
                "current_node": numa.get_mempolicy()[1],
                "topology": self.numa_topology
            }
        except Exception as e:
            return {"available": False, "error": str(e)}
    
    async def monitor_resource_usage(self) -> Dict[str, Any]:
        """Monitor current resource usage."""
        try:
            process = psutil.Process()
            
            # CPU usage
            cpu_percent = process.cpu_percent(interval=0.1)
            
            # Memory usage
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # I/O stats
            try:
                io_counters = process.io_counters()
                io_stats = {
                    "read_count": io_counters.read_count,
                    "write_count": io_counters.write_count,
                    "read_bytes": io_counters.read_bytes,
                    "write_bytes": io_counters.write_bytes
                }
            except (AttributeError, OSError):
                io_stats = {}
            
            return {
                "cpu_percent": cpu_percent,
                "memory_rss_mb": memory_info.rss / (1024 * 1024),
                "memory_vms_mb": memory_info.vms / (1024 * 1024),
                "memory_percent": memory_percent,
                "num_threads": process.num_threads(),
                "num_fds": process.num_fds() if hasattr(process, 'num_fds') else None,
                "io_stats": io_stats
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def cleanup(self):
        """Restore original system settings."""
        try:
            if not self.optimization_applied:
                return
            
            process = psutil.Process()
            
            # Restore CPU affinity
            if self.original_affinity is not None:
                try:
                    process.cpu_affinity(self.original_affinity)
                except (AttributeError, OSError):
                    pass
            
            # Restore process priority
            if self.original_priority is not None:
                try:
                    process.nice(self.original_priority)
                except (AttributeError, OSError):
                    pass
            
            # Reset NUMA policy
            if NUMA_AVAILABLE and self.numa_topology:
                try:
                    numa.set_mempolicy(numa.MPOL_DEFAULT)
                except Exception:
                    pass
            
        except Exception as e:
            print(f"Warning: Failed to restore some system settings: {e}")


def set_thread_affinity(thread_id: int, cpu_list: List[int]):
    """Set CPU affinity for a specific thread (Linux only)."""
    try:
        import ctypes
        import ctypes.util
        
        # This is Linux-specific
        if os.name != 'posix':
            return
        
        libc = ctypes.CDLL(ctypes.util.find_library('c'))
        
        # Create CPU set
        cpu_set_t = ctypes.c_ulong
        cpu_set = cpu_set_t(0)
        
        for cpu in cpu_list:
            cpu_set |= (1 << cpu)
        
        # Set affinity
        libc.sched_setaffinity(thread_id, ctypes.sizeof(cpu_set_t), ctypes.byref(cpu_set))
        
    except Exception:
        pass  # Thread affinity not available or failed


class ResourceMonitor:
    """Continuous resource monitoring for optimization feedback."""
    
    def __init__(self, check_interval: float = 1.0):
        self.check_interval = check_interval
        self.monitoring = False
        self.stats_history = []
        self.max_history = 1000
    
    async def start_monitoring(self):
        """Start continuous resource monitoring."""
        self.monitoring = True
        
        while self.monitoring:
            try:
                stats = await self._collect_stats()
                self.stats_history.append(stats)
                
                # Limit history size
                if len(self.stats_history) > self.max_history:
                    self.stats_history.pop(0)
                
                await asyncio.sleep(self.check_interval)
                
            except Exception:
                await asyncio.sleep(self.check_interval)
    
    async def _collect_stats(self) -> Dict[str, Any]:
        """Collect current resource statistics."""
        import time
        
        stats = {
            "timestamp": time.time(),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_io": psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
            "network_io": psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
        }
        
        return stats
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
    
    def get_resource_trends(self) -> Dict[str, Any]:
        """Get resource usage trends."""
        if len(self.stats_history) < 2:
            return {}
        
        recent_stats = self.stats_history[-10:]  # Last 10 samples
        
        cpu_trend = [s["cpu_percent"] for s in recent_stats]
        memory_trend = [s["memory_percent"] for s in recent_stats]
        
        return {
            "cpu_average": sum(cpu_trend) / len(cpu_trend),
            "cpu_max": max(cpu_trend),
            "memory_average": sum(memory_trend) / len(memory_trend),
            "memory_max": max(memory_trend),
            "sample_count": len(recent_stats)
        }
