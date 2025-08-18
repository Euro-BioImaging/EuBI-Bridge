"""
Configuration Profile Manager for High-Performance BigTIFF Converter
Provides optimized configuration profiles for different hardware environments.
"""

from typing import Dict, Any


class ConfigProfileManager:
    """Manages configuration profiles for different hardware environments."""
    
    def __init__(self):
        self.profiles = {
            "hpc": {
                "description": "High-Performance Computing cluster with fast interconnect",
                "workers": "auto",  # Will be determined by hardware detection
                "chunk_memory_mb": 512,
                "compression": "blosc2-lz4",
                "compression_level": 3,
                "use_numa": True,
                "use_async_io": True,
                "pyramid_levels": None,  # Auto-calculate
                "downsample_factor": 2,
                "memory_mapping": True,
                "prefetch_chunks": 4,
                "cache_size_mb": 1024,
                "parallel_io": True,
                "optimize_for": "throughput"
            },
            
            "workstation": {
                "description": "High-end workstation with fast storage and plenty of RAM",
                "workers": "auto",
                "chunk_memory_mb": 256,
                "compression": "blosc2-lz4",
                "compression_level": 3,
                "use_numa": False,
                "use_async_io": True,
                "pyramid_levels": None,
                "downsample_factor": 2,
                "memory_mapping": True,
                "prefetch_chunks": 2,
                "cache_size_mb": 512,
                "parallel_io": True,
                "optimize_for": "balanced"
            },
            
            "cloud": {
                "description": "Cloud computing instance with variable performance",
                "workers": "auto",
                "chunk_memory_mb": 128,
                "compression": "blosc2-zstd",
                "compression_level": 2,
                "use_numa": False,
                "use_async_io": True,
                "pyramid_levels": None,
                "downsample_factor": 2,
                "memory_mapping": False,  # May have slower storage
                "prefetch_chunks": 1,
                "cache_size_mb": 256,
                "parallel_io": False,
                "optimize_for": "memory"
            },
            
            "constrained": {
                "description": "Resource-constrained environment with limited RAM/CPU",
                "workers": 2,
                "chunk_memory_mb": 64,
                "compression": "lz4",  # Faster, less CPU intensive
                "compression_level": 1,
                "use_numa": False,
                "use_async_io": False,
                "pyramid_levels": 3,  # Fewer levels to save space
                "downsample_factor": 4,  # Higher downsampling
                "memory_mapping": True,  # Essential for limited RAM
                "prefetch_chunks": 0,
                "cache_size_mb": 64,
                "parallel_io": False,
                "optimize_for": "memory"
            }
        }
    
    def get_profile_config(self, profile_name: str) -> Dict[str, Any]:
        """Get configuration for a specific profile."""
        profile_name = profile_name.lower()
        if profile_name not in self.profiles:
            raise ValueError(f"Unknown profile: {profile_name}. Available profiles: {list(self.profiles.keys())}")
        
        return self.profiles[profile_name].copy()
    
    def get_all_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get all available profiles."""
        return self.profiles.copy()
    
    def add_custom_profile(self, name: str, config: Dict[str, Any]):
        """Add a custom configuration profile."""
        self.profiles[name.lower()] = config
    
    def get_profile_names(self) -> list:
        """Get list of available profile names."""
        return list(self.profiles.keys())
    
    def recommend_profile_for_hardware(self, hardware_info: Dict[str, Any]) -> str:
        """Recommend a profile based on hardware information."""
        cpu_cores = hardware_info.get("cpu_info", {}).get("logical_cores", 1)
        memory_gb = hardware_info.get("memory_info", {}).get("total_gb", 0)
        storage_type = hardware_info.get("storage_info", {}).get("type", "Unknown")
        numa_available = hardware_info.get("numa", {}).get("available", False)
        
        # Decision logic based on hardware capabilities
        if cpu_cores >= 16 and memory_gb >= 32 and numa_available:
            return "hpc"
        elif cpu_cores >= 8 and memory_gb >= 16 and "SSD" in storage_type:
            return "workstation"
        elif memory_gb >= 8:
            return "cloud"
        else:
            return "constrained"
    
    def validate_profile_config(self, config: Dict[str, Any]) -> bool:
        """Validate a profile configuration."""
        required_fields = ["workers", "chunk_memory_mb", "compression", "use_numa", "use_async_io"]
        
        for field in required_fields:
            if field not in config:
                return False
        
        # Validate compression options
        valid_compressions = ["blosc2-lz4", "blosc2-zstd", "lz4", "none"]
        if config["compression"] not in valid_compressions:
            return False
        
        # Validate numeric ranges
        if isinstance(config["chunk_memory_mb"], int) and config["chunk_memory_mb"] <= 0:
            return False
        
        return True
    
    def merge_with_profile(self, base_config: Dict[str, Any], profile_name: str) -> Dict[str, Any]:
        """Merge user configuration with profile defaults."""
        profile_config = self.get_profile_config(profile_name)
        
        # Start with profile defaults
        merged_config = profile_config.copy()
        
        # Override with user-specified values
        for key, value in base_config.items():
            if value is not None:  # Only override if user explicitly set a value
                merged_config[key] = value
        
        return merged_config