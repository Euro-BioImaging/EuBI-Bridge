"""
Hardware detection and optimization recommendations.
Analyzes system capabilities and suggests optimal configurations.
"""

import os
import platform
import psutil
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
import subprocess
import json


class HardwareDetector:
    """Comprehensive hardware detection and analysis."""
    
    def __init__(self):
        self.hardware_info = {}
        self.optimization_recommendations = {}
        self.detected = False
    
    async def detect_hardware(self):
        """Perform comprehensive hardware detection."""
        if self.detected:
            return self.hardware_info
        
        self.hardware_info = {
            "cpu": await self._detect_cpu(),
            "memory": await self._detect_memory(),
            "storage": await self._detect_storage(),
            "numa": await self._detect_numa(),
            "network": await self._detect_network(),
            "platform": await self._detect_platform(),
            "capabilities": await self._detect_capabilities()
        }
        
        # Generate optimization recommendations
        self.optimization_recommendations = await self._generate_recommendations()
        self.detected = True
        
        return self.hardware_info
    
    async def _detect_cpu(self) -> Dict[str, Any]:
        """Detect CPU specifications and capabilities."""
        cpu_info = {
            "model": "Unknown",
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
            "architecture": platform.machine(),
            "frequency": {},
            "features": [],
            "cache": {},
            "recommended_workers": psutil.cpu_count()
        }
        
        # CPU frequency information
        try:
            freq = psutil.cpu_freq()
            if freq:
                cpu_info["frequency"] = {
                    "min_mhz": freq.min,
                    "max_mhz": freq.max,
                    "current_mhz": freq.current
                }
        except (AttributeError, OSError):
            pass
        
        # CPU model and features (Linux)
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()
                
                # Extract model name
                for line in cpuinfo.split("\n"):
                    if line.startswith("model name"):
                        cpu_info["model"] = line.split(":", 1)[1].strip()
                        break
                
                # Extract features
                for line in cpuinfo.split("\n"):
                    if line.startswith("flags"):
                        flags = line.split(":", 1)[1].strip().split()
                        # Focus on performance-relevant features
                        relevant_features = ["avx", "avx2", "avx512f", "sse4_1", "sse4_2", 
                                           "fma", "aes", "sha_ni", "bmi1", "bmi2"]
                        cpu_info["features"] = [f for f in relevant_features if f in flags]
                        break
                
                # Try to get cache information
                cpu_info["cache"] = await self._detect_cpu_cache_linux()
                
            except (IOError, OSError):
                pass
        
        # macOS CPU detection
        elif platform.system() == "Darwin":
            try:
                # Get CPU model
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], 
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    cpu_info["model"] = result.stdout.strip()
                
                # Get CPU features
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.features"], 
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    features = result.stdout.strip().split()
                    cpu_info["features"] = [f.lower() for f in features]
                
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                pass
        
        # Windows CPU detection
        elif platform.system() == "Windows":
            try:
                import winreg
                
                # Get CPU model from registry
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                  r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                    cpu_info["model"] = winreg.QueryValueEx(key, "ProcessorNameString")[0]
                
            except (ImportError, OSError):
                pass
        
        # Optimize worker count based on workload type
        if cpu_info["threads"] > cpu_info["cores"]:
            # Hyperthreading available - good for I/O bound tasks
            cpu_info["recommended_workers"] = cpu_info["threads"]
        else:
            # No hyperthreading - use core count for CPU bound tasks
            cpu_info["recommended_workers"] = cpu_info["cores"]
        
        return cpu_info
    
    async def _detect_cpu_cache_linux(self) -> Dict[str, Any]:
        """Detect CPU cache information on Linux."""
        cache_info = {}
        
        try:
            cache_dir = Path("/sys/devices/system/cpu/cpu0/cache")
            if cache_dir.exists():
                for index_dir in cache_dir.glob("index*"):
                    try:
                        with open(index_dir / "level") as f:
                            level = f.read().strip()
                        with open(index_dir / "size") as f:
                            size = f.read().strip()
                        with open(index_dir / "type") as f:
                            cache_type = f.read().strip()
                        
                        cache_info[f"L{level}_{cache_type.lower()}"] = size
                    except (IOError, OSError):
                        continue
        except OSError:
            pass
        
        return cache_info
    
    async def _detect_memory(self) -> Dict[str, Any]:
        """Detect memory specifications and characteristics."""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        memory_info = {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "swap_total_gb": swap.total / (1024**3),
            "type": "Unknown",
            "speed_mhz": None,
            "recommended_chunk_mb": self._calculate_optimal_chunk_size(memory.total)
        }
        
        # Platform-specific memory detection
        if platform.system() == "Linux":
            memory_info.update(await self._detect_memory_linux())
        elif platform.system() == "Darwin":
            memory_info.update(await self._detect_memory_macos())
        elif platform.system() == "Windows":
            memory_info.update(await self._detect_memory_windows())
        
        return memory_info
    
    async def _detect_memory_linux(self) -> Dict[str, Any]:
        """Detect memory details on Linux."""
        info = {}
        
        try:
            # Memory type and speed from dmidecode (requires root)
            try:
                result = subprocess.run(
                    ["dmidecode", "-t", "memory"], 
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    output = result.stdout
                    
                    # Extract memory type
                    for line in output.split("\n"):
                        if "Type:" in line and "DDR" in line:
                            info["type"] = line.split(":", 1)[1].strip()
                            break
                    
                    # Extract memory speed
                    for line in output.split("\n"):
                        if "Speed:" in line and "MHz" in line:
                            speed_str = line.split(":", 1)[1].strip()
                            if "MHz" in speed_str:
                                try:
                                    info["speed_mhz"] = int(speed_str.split()[0])
                                    break
                                except (ValueError, IndexError):
                                    pass
            except (subprocess.TimeoutExpired, subprocess.SubprocessError, PermissionError):
                pass
            
            # Memory information from /proc/meminfo
            with open("/proc/meminfo", "r") as f:
                meminfo = f.read()
            
            for line in meminfo.split("\n"):
                if line.startswith("Hugepagesize:"):
                    try:
                        hugepage_kb = int(line.split()[1])
                        info["hugepage_size_mb"] = hugepage_kb / 1024
                    except (ValueError, IndexError):
                        pass
                
        except (IOError, OSError):
            pass
        
        return info
    
    async def _detect_memory_macos(self) -> Dict[str, Any]:
        """Detect memory details on macOS."""
        info = {}
        
        try:
            # Memory type and speed
            result = subprocess.run(
                ["system_profiler", "SPMemoryDataType", "-json"], 
                capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout)
                    memory_data = data.get("SPMemoryDataType", [])
                    if memory_data:
                        first_slot = memory_data[0]
                        info["type"] = first_slot.get("dimm_type", "Unknown")
                        speed_str = first_slot.get("dimm_speed", "")
                        if "MHz" in speed_str:
                            try:
                                info["speed_mhz"] = int(speed_str.split()[0])
                            except (ValueError, IndexError):
                                pass
                except (json.JSONDecodeError, KeyError):
                    pass
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            pass
        
        return info
    
    async def _detect_memory_windows(self) -> Dict[str, Any]:
        """Detect memory details on Windows."""
        info = {}
        
        try:
            # Use wmic to get memory information
            result = subprocess.run(
                ["wmic", "memorychip", "get", "MemoryType,Speed", "/format:csv"], 
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                for line in lines[1:]:  # Skip header
                    if line.strip():
                        parts = line.split(",")
                        if len(parts) >= 3:
                            try:
                                memory_type = int(parts[1])
                                speed = int(parts[2])
                                
                                # Memory type mapping (simplified)
                                type_map = {20: "DDR", 21: "DDR2", 24: "DDR3", 26: "DDR4", 30: "DDR5"}
                                info["type"] = type_map.get(memory_type, f"Type_{memory_type}")
                                info["speed_mhz"] = speed
                                break
                            except (ValueError, IndexError):
                                continue
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            pass
        
        return info
    
    def _calculate_optimal_chunk_size(self, total_memory: int) -> int:
        """Calculate optimal chunk size based on available memory."""
        total_gb = total_memory / (1024**3)
        
        if total_gb >= 64:
            return 512  # 512MB chunks for high-memory systems
        elif total_gb >= 32:
            return 256  # 256MB chunks for medium-memory systems
        elif total_gb >= 16:
            return 128  # 128MB chunks for standard systems
        else:
            return 64   # 64MB chunks for low-memory systems
    
    async def _detect_storage(self) -> Dict[str, Any]:
        """Detect storage specifications and performance characteristics."""
        storage_info = {
            "devices": [],
            "type": "Unknown",
            "estimated_speed": "Unknown",
            "recommended_buffer_mb": 64
        }
        
        # Get disk usage for main partition
        try:
            disk_usage = psutil.disk_usage("/")
            storage_info["main_partition"] = {
                "total_gb": disk_usage.total / (1024**3),
                "free_gb": disk_usage.free / (1024**3),
                "used_gb": disk_usage.used / (1024**3)
            }
        except (OSError, AttributeError):
            pass
        
        # Platform-specific storage detection
        if platform.system() == "Linux":
            storage_info.update(await self._detect_storage_linux())
        elif platform.system() == "Darwin":
            storage_info.update(await self._detect_storage_macos())
        elif platform.system() == "Windows":
            storage_info.update(await self._detect_storage_windows())
        
        # Set recommended buffer based on storage type
        if "ssd" in storage_info["type"].lower() or "nvme" in storage_info["type"].lower():
            storage_info["recommended_buffer_mb"] = 128  # Larger buffer for SSDs
        else:
            storage_info["recommended_buffer_mb"] = 64   # Smaller buffer for HDDs
        
        return storage_info
    
    async def _detect_storage_linux(self) -> Dict[str, Any]:
        """Detect storage details on Linux."""
        info = {"devices": []}
        
        try:
            # Get block device information
            result = subprocess.run(
                ["lsblk", "-J", "-o", "NAME,TYPE,SIZE,MOUNTPOINT,ROTA"], 
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout)
                    for device in data.get("blockdevices", []):
                        if device.get("type") == "disk":
                            device_info = {
                                "name": device.get("name"),
                                "size": device.get("size"),
                                "rotational": device.get("rota") == "1",
                                "type": "HDD" if device.get("rota") == "1" else "SSD"
                            }
                            info["devices"].append(device_info)
                            
                            # Set main storage type based on root device
                            if device.get("children"):
                                for child in device["children"]:
                                    if child.get("mountpoint") == "/":
                                        info["type"] = device_info["type"]
                                        break
                except (json.JSONDecodeError, KeyError):
                    pass
            
            # Try to detect NVMe drives
            try:
                result = subprocess.run(
                    ["ls", "/dev/nvme*n1"], 
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    info["type"] = "NVMe SSD"
                    info["estimated_speed"] = "High (NVMe)"
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                pass
            
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            pass
        
        return info
    
    async def _detect_storage_macos(self) -> Dict[str, Any]:
        """Detect storage details on macOS."""
        info = {}
        
        try:
            result = subprocess.run(
                ["system_profiler", "SPStorageDataType", "-json"], 
                capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout)
                    storage_data = data.get("SPStorageDataType", [])
                    if storage_data:
                        main_drive = storage_data[0]
                        
                        # Detect SSD vs HDD
                        medium_type = main_drive.get("spstorage_medium_type", "").lower()
                        if "ssd" in medium_type or "flash" in medium_type:
                            info["type"] = "SSD"
                            info["estimated_speed"] = "High (SSD)"
                        else:
                            info["type"] = "HDD"
                            info["estimated_speed"] = "Standard (HDD)"
                except (json.JSONDecodeError, KeyError):
                    pass
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            pass
        
        return info
    
    async def _detect_storage_windows(self) -> Dict[str, Any]:
        """Detect storage details on Windows."""
        info = {}
        
        try:
            # Use wmic to get disk drive information
            result = subprocess.run(
                ["wmic", "diskdrive", "get", "MediaType,Size", "/format:csv"], 
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                for line in lines[1:]:  # Skip header
                    if line.strip():
                        parts = line.split(",")
                        if len(parts) >= 3:
                            media_type = parts[1].strip().lower()
                            if "ssd" in media_type or "solid state" in media_type:
                                info["type"] = "SSD"
                                info["estimated_speed"] = "High (SSD)"
                                break
                            elif "fixed" in media_type:
                                info["type"] = "HDD"
                                info["estimated_speed"] = "Standard (HDD)"
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            pass
        
        return info
    
    async def _detect_numa(self) -> Dict[str, Any]:
        """Detect NUMA topology and capabilities."""
        numa_info = {"available": False}
        
        try:
            import numa
            if numa.available():
                numa_info = {
                    "available": True,
                    "nodes": numa.get_max_node() + 1,
                    "current_node": numa.get_mempolicy()[1],
                    "node_details": []
                }
                
                for node in range(numa_info["nodes"]):
                    try:
                        node_cpus = numa.node_to_cpus(node)
                        node_memory = numa.node_size(node) / (1024**3)  # GB
                        
                        numa_info["node_details"].append({
                            "node": node,
                            "cpus": node_cpus,
                            "memory_gb": node_memory
                        })
                    except Exception:
                        continue
        except ImportError:
            # Check if NUMA is available through system calls
            if platform.system() == "Linux":
                try:
                    result = subprocess.run(
                        ["numactl", "--hardware"], 
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        numa_info["available"] = True
                        # Parse numactl output for basic info
                        lines = result.stdout.split("\n")
                        for line in lines:
                            if "available:" in line and "nodes" in line:
                                try:
                                    numa_info["nodes"] = int(line.split()[1])
                                except (ValueError, IndexError):
                                    pass
                except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
                    # numactl not available, NUMA not supported
                    numa_info["available"] = False
        
        return numa_info
    
    async def _detect_network(self) -> Dict[str, Any]:
        """Detect network interfaces and capabilities."""
        network_info = {"interfaces": []}
        
        try:
            interfaces = psutil.net_if_addrs()
            for interface_name, addresses in interfaces.items():
                if interface_name.startswith(("lo", "127")):  # Skip loopback
                    continue
                
                interface_info = {"name": interface_name, "addresses": []}
                for addr in addresses:
                    if addr.family.name in ["AF_INET", "AF_INET6"]:
                        interface_info["addresses"].append({
                            "family": addr.family.name,
                            "address": addr.address
                        })
                
                if interface_info["addresses"]:
                    network_info["interfaces"].append(interface_info)
        except Exception:
            pass
        
        return network_info
    
    async def _detect_platform(self) -> Dict[str, Any]:
        """Detect platform and OS details."""
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation()
        }
    
    async def _detect_capabilities(self) -> Dict[str, Any]:
        """Detect system capabilities and features."""
        capabilities = {
            "containers": False,
            "virtualization": False,
            "hardware_acceleration": []
        }
        
        # Check for container environments
        if os.path.exists("/.dockerenv") or os.environ.get("container") == "docker":
            capabilities["containers"] = True
        
        # Check for virtualization (basic heuristics)
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()
                if "hypervisor" in cpuinfo:
                    capabilities["virtualization"] = True
            except (IOError, OSError):
                pass
        
        # Check for hardware acceleration
        cpu_features = self.hardware_info.get("cpu", {}).get("features", [])
        if "avx2" in cpu_features:
            capabilities["hardware_acceleration"].append("AVX2")
        if "avx512f" in cpu_features:
            capabilities["hardware_acceleration"].append("AVX512")
        
        return capabilities
    
    async def _generate_recommendations(self) -> Dict[str, Any]:
        """Generate optimization recommendations based on detected hardware."""
        recommendations = {
            "profile": "auto",
            "workers": self.hardware_info["cpu"]["recommended_workers"],
            "chunk_memory_mb": self.hardware_info["memory"]["recommended_chunk_mb"],
            "compression": "blosc2-lz4",
            "use_numa": self.hardware_info["numa"]["available"],
            "io_buffer_mb": self.hardware_info["storage"]["recommended_buffer_mb"],
            "optimizations": []
        }
        
        # Determine optimal profile
        memory_gb = self.hardware_info["memory"]["total_gb"]
        cpu_cores = self.hardware_info["cpu"]["cores"]
        
        if memory_gb >= 64 and cpu_cores >= 16:
            recommendations["profile"] = "hpc"
            recommendations["optimizations"].append("High-memory HPC configuration")
        elif memory_gb >= 32 and cpu_cores >= 8:
            recommendations["profile"] = "workstation"
            recommendations["optimizations"].append("Workstation configuration")
        elif memory_gb >= 16:
            recommendations["profile"] = "standard"
            recommendations["optimizations"].append("Standard configuration")
        else:
            recommendations["profile"] = "memory-constrained"
            recommendations["optimizations"].append("Memory-constrained configuration")
            recommendations["chunk_memory_mb"] = min(recommendations["chunk_memory_mb"], 64)
        
        # Storage optimizations
        storage_type = self.hardware_info["storage"]["type"].lower()
        if "nvme" in storage_type or "ssd" in storage_type:
            recommendations["optimizations"].append("SSD optimization enabled")
            recommendations["io_buffer_mb"] = max(recommendations["io_buffer_mb"], 128)
        
        # NUMA optimizations
        if self.hardware_info["numa"]["available"]:
            recommendations["optimizations"].append("NUMA-aware processing enabled")
            numa_nodes = self.hardware_info["numa"].get("nodes", 1)
            if numa_nodes > 1:
                recommendations["workers"] = min(recommendations["workers"], cpu_cores)
        
        # CPU feature optimizations
        cpu_features = self.hardware_info["cpu"].get("features", [])
        if "avx2" in cpu_features:
            recommendations["optimizations"].append("AVX2 acceleration available")
        if "avx512f" in cpu_features:
            recommendations["optimizations"].append("AVX512 acceleration available")
        
        # Memory optimization
        if memory_gb < 8:
            recommendations["optimizations"].append("Memory usage optimization required")
            recommendations["workers"] = min(recommendations["workers"], 2)
        
        return recommendations
    
    def get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information."""
        return self.hardware_info.get("cpu", {})
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory information."""
        return self.hardware_info.get("memory", {})
    
    def recommend_profile(self) -> str:
        """Recommend optimization profile based on hardware."""
        if not self.detected:
            return "cloud"  # Default fallback
        
        memory_gb = self.hardware_info["memory"]["total_gb"]
        cpu_cores = self.hardware_info["cpu"]["cores"]
        numa_available = self.hardware_info["numa"]["available"]
        storage_type = self.hardware_info["storage"]["type"]
        
        # Decision logic based on hardware capabilities
        if cpu_cores >= 16 and memory_gb >= 32 and numa_available:
            return "hpc"
        elif cpu_cores >= 8 and memory_gb >= 16 and "SSD" in storage_type:
            return "workstation"
        elif memory_gb >= 8:
            return "cloud"
        else:
            return "constrained"
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage information."""
        return self.hardware_info.get("storage", {})
    
    def get_numa_info(self) -> Dict[str, Any]:
        """Get NUMA information."""
        return self.hardware_info.get("numa", {})
    
    def get_recommended_profile(self) -> str:
        """Get recommended configuration profile."""
        return self.optimization_recommendations.get("profile", "auto")
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get comprehensive optimization recommendations."""
        return self.optimization_recommendations
    
    async def generate_system_report(self, output_file: Optional[str] = None) -> str:
        """Generate a detailed system report."""
        if not self.detected:
            await self.detect_hardware()
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("System Hardware Report")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Platform information
        platform_info = self.hardware_info["platform"]
        report_lines.append("Platform Information:")
        report_lines.append(f"  OS: {platform_info['system']} {platform_info['release']}")
        report_lines.append(f"  Architecture: {platform_info['machine']}")
        report_lines.append(f"  Python: {platform_info['python_version']} ({platform_info['python_implementation']})")
        report_lines.append("")
        
        # CPU information
        cpu_info = self.hardware_info["cpu"]
        report_lines.append("CPU Information:")
        report_lines.append(f"  Model: {cpu_info['model']}")
        report_lines.append(f"  Cores: {cpu_info['cores']} physical, {cpu_info['threads']} logical")
        if cpu_info.get("frequency"):
            freq = cpu_info["frequency"]
            report_lines.append(f"  Frequency: {freq.get('current_mhz', 'Unknown')} MHz (max: {freq.get('max_mhz', 'Unknown')} MHz)")
        if cpu_info.get("features"):
            report_lines.append(f"  Features: {', '.join(cpu_info['features'])}")
        report_lines.append("")
        
        # Memory information
        memory_info = self.hardware_info["memory"]
        report_lines.append("Memory Information:")
        report_lines.append(f"  Total: {memory_info['total_gb']:.2f} GB")
        report_lines.append(f"  Available: {memory_info['available_gb']:.2f} GB")
        if memory_info.get("type"):
            report_lines.append(f"  Type: {memory_info['type']}")
        if memory_info.get("speed_mhz"):
            report_lines.append(f"  Speed: {memory_info['speed_mhz']} MHz")
        report_lines.append("")
        
        # Storage information
        storage_info = self.hardware_info["storage"]
        report_lines.append("Storage Information:")
        report_lines.append(f"  Type: {storage_info['type']}")
        report_lines.append(f"  Estimated Speed: {storage_info['estimated_speed']}")
        if storage_info.get("main_partition"):
            mp = storage_info["main_partition"]
            report_lines.append(f"  Main Partition: {mp['free_gb']:.2f} GB free of {mp['total_gb']:.2f} GB")
        report_lines.append("")
        
        # NUMA information
        numa_info = self.hardware_info["numa"]
        report_lines.append("NUMA Information:")
        if numa_info["available"]:
            report_lines.append(f"  Available: Yes ({numa_info.get('nodes', 'Unknown')} nodes)")
        else:
            report_lines.append("  Available: No")
        report_lines.append("")
        
        # Optimization recommendations
        recommendations = self.optimization_recommendations
        report_lines.append("Optimization Recommendations:")
        report_lines.append(f"  Recommended Profile: {recommendations['profile']}")
        report_lines.append(f"  Workers: {recommendations['workers']}")
        report_lines.append(f"  Chunk Memory: {recommendations['chunk_memory_mb']} MB")
        report_lines.append(f"  Compression: {recommendations['compression']}")
        report_lines.append(f"  Use NUMA: {recommendations['use_numa']}")
        report_lines.append(f"  I/O Buffer: {recommendations['io_buffer_mb']} MB")
        
        if recommendations.get("optimizations"):
            report_lines.append("  Available Optimizations:")
            for opt in recommendations["optimizations"]:
                report_lines.append(f"    - {opt}")
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
        
        return report


# Utility functions
async def detect_hardware_quick() -> Dict[str, Any]:
    """Quick hardware detection for immediate use."""
    detector = HardwareDetector()
    return await detector.detect_hardware()


async def get_optimization_recommendations() -> Dict[str, Any]:
    """Get optimization recommendations for current system."""
    detector = HardwareDetector()
    await detector.detect_hardware()
    return detector.get_optimization_recommendations()
