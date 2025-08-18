"""
High-performance compression engine with optimized codecs.
Supports Blosc2, LZ4, and other fast compression algorithms.
"""

import numpy as np
from typing import Dict, Any, Optional, Union
import time

try:
    import blosc2
    BLOSC2_AVAILABLE = True
except ImportError:
    BLOSC2_AVAILABLE = False

try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

try:
    from numcodecs import Blosc, LZ4
    NUMCODECS_AVAILABLE = True
except ImportError:
    NUMCODECS_AVAILABLE = False


class CompressionEngine:
    """High-performance compression engine with multiple codec support."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.compression_type = config.get("compression", "blosc2-lz4")
        self.compression_level = config.get("compression_level", 3)
        self.shuffle = config.get("shuffle", True)
        
        # Performance tracking
        self.compression_stats = {
            "total_compressed": 0,
            "total_uncompressed": 0,
            "compression_time": 0.0,
            "decompression_time": 0.0
        }
        
        # Initialize compressor
        self.compressor = self._create_compressor()
    
    def _create_compressor(self) -> Optional[Any]:
        """Create optimal compressor based on configuration and available libraries."""
        compression_type = self.compression_type.lower()
        
        # Always prefer numcodecs for Zarr compatibility
        if compression_type.startswith("blosc") and NUMCODECS_AVAILABLE:
            return self._create_blosc_compressor()
        elif compression_type == "lz4" and NUMCODECS_AVAILABLE:
            return LZ4(acceleration=1)
        elif compression_type == "none":
            return None
        else:
            # Fallback to best available
            return self._create_fallback_compressor()
    
    def _create_blosc2_lz4_compressor(self):
        """Create Blosc2 compressor with LZ4 for maximum speed."""
        # Use numcodecs constants for shuffle
        try:
            shuffle_mode = 1 if self.shuffle else 0  # 1=SHUFFLE, 0=NOSHUFFLE
        except:
            shuffle_mode = 1  # Default to shuffle
        
        return Blosc2Compressor(
            cname="lz4",
            clevel=self.compression_level,
            shuffle=shuffle_mode,
            blocksize=0,  # Auto block size
            nthreads=self.config.get("compression_threads", 0)  # Auto threads
        )
    
    def _create_blosc2_zstd_compressor(self):
        """Create Blosc2 compressor with Zstandard for better compression."""
        # Use numcodecs constants for shuffle
        try:
            shuffle_mode = 1 if self.shuffle else 0  # 1=SHUFFLE, 0=NOSHUFFLE
        except:
            shuffle_mode = 1  # Default to shuffle
        
        return Blosc2Compressor(
            cname="zstd",
            clevel=self.compression_level,
            shuffle=shuffle_mode,
            blocksize=0,
            nthreads=self.config.get("compression_threads", 0)
        )
    
    def _create_lz4_compressor(self):
        """Create standalone LZ4 compressor."""
        return LZ4Compressor(
            compression_level=self.compression_level
        )
    
    def _create_blosc_compressor(self):
        """Create Blosc compressor using numcodecs."""
        if not NUMCODECS_AVAILABLE:
            return None
            
        shuffle_mode = Blosc.SHUFFLE if self.shuffle else Blosc.NOSHUFFLE
        
        cname = "lz4"
        if "zstd" in self.compression_type:
            cname = "zstd"
        elif "lz4hc" in self.compression_type:
            cname = "lz4hc"
        
        return Blosc(
            cname=cname,
            clevel=self.compression_level,
            shuffle=shuffle_mode,
            blocksize=0
        )
    
    def _create_fallback_compressor(self):
        """Create fallback compressor when preferred option not available."""
        if NUMCODECS_AVAILABLE:
            return Blosc(cname="lz4", clevel=1, shuffle=Blosc.SHUFFLE)
        return None
    
    def get_compressor(self):
        """Get the configured compressor for Zarr."""
        # Return numcodecs compressor for Zarr compatibility
        if self.compressor is None:
            return None
        elif hasattr(self.compressor, 'codec'):
            return self.compressor.codec
        elif NUMCODECS_AVAILABLE:
            # Create a standard numcodecs compressor as fallback
            return Blosc(cname="lz4", clevel=self.compression_level, shuffle=Blosc.SHUFFLE)
        else:
            return None
    
    def compress_array(self, data: np.ndarray) -> bytes:
        """Compress numpy array and return compressed bytes."""
        if self.compressor is None:
            return data.tobytes()
        
        start_time = time.time()
        
        try:
            if hasattr(self.compressor, 'compress'):
                # Direct compression
                compressed = self.compressor.compress(data)
            else:
                # Fallback to encoding
                compressed = self.compressor.encode(data)
            
            # Update statistics
            self.compression_stats["total_uncompressed"] += data.nbytes
            self.compression_stats["total_compressed"] += len(compressed)
            self.compression_stats["compression_time"] += time.time() - start_time
            
            return compressed
            
        except Exception as e:
            # Fallback to uncompressed
            return data.tobytes()
    
    def decompress_array(self, compressed_data: bytes, shape: tuple, 
                        dtype: np.dtype) -> np.ndarray:
        """Decompress bytes back to numpy array."""
        if self.compressor is None:
            return np.frombuffer(compressed_data, dtype=dtype).reshape(shape)
        
        start_time = time.time()
        
        try:
            if hasattr(self.compressor, 'decompress'):
                # Direct decompression
                decompressed = self.compressor.decompress(compressed_data)
                array = np.frombuffer(decompressed, dtype=dtype).reshape(shape)
            else:
                # Fallback to decoding
                array = self.compressor.decode(compressed_data)
            
            self.compression_stats["decompression_time"] += time.time() - start_time
            return array
            
        except Exception as e:
            # Fallback to uncompressed
            return np.frombuffer(compressed_data, dtype=dtype).reshape(shape)
    
    def get_estimated_compression_ratio(self) -> float:
        """Get estimated compression ratio based on compression type and data."""
        # Default ratios based on typical microscopy data
        ratios = {
            "blosc2-lz4": 2.5,
            "blosc2-zstd": 4.0,
            "lz4": 2.0,
            "blosc-lz4": 2.5,
            "blosc-zstd": 4.0,
            "none": 1.0
        }
        
        return ratios.get(self.compression_type.lower(), 2.0)
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression performance statistics."""
        stats = self.compression_stats.copy()
        
        if stats["total_uncompressed"] > 0:
            stats["actual_compression_ratio"] = (
                stats["total_uncompressed"] / max(stats["total_compressed"], 1)
            )
        else:
            stats["actual_compression_ratio"] = 1.0
        
        if stats["compression_time"] > 0:
            stats["compression_throughput_mb_s"] = (
                (stats["total_uncompressed"] / (1024 * 1024)) / stats["compression_time"]
            )
        else:
            stats["compression_throughput_mb_s"] = 0.0
        
        return stats
    
    def optimize_for_data_type(self, dtype: np.dtype, data_characteristics: Dict[str, Any]):
        """Optimize compression settings based on data type and characteristics."""
        # Adjust settings based on data type
        if dtype == np.uint8:
            # 8-bit data: optimize for speed
            if "lz4" not in self.compression_type:
                self.compression_type = "blosc2-lz4"
                self.compression_level = min(self.compression_level, 3)
        elif dtype in [np.uint16, np.int16]:
            # 16-bit data: balance speed and compression
            self.compression_level = min(self.compression_level, 5)
        elif dtype in [np.float32, np.float64]:
            # Float data: may benefit from better compression
            if data_characteristics.get("has_many_zeros", False):
                self.shuffle = True
        
        # Recreate compressor with new settings
        self.compressor = self._create_compressor()


class Blosc2Compressor:
    """High-performance Blosc2 compressor wrapper."""
    
    def __init__(self, cname: str = "lz4", clevel: int = 3, 
                 shuffle: int = None, blocksize: int = 0, nthreads: int = 0):
        if not BLOSC2_AVAILABLE:
            raise ImportError("Blosc2 not available")
        
        self.cname = cname
        self.clevel = clevel
        self.shuffle = shuffle or blosc2.SHUFFLE
        self.blocksize = blocksize
        self.nthreads = nthreads
        
        # Set global threads if specified
        if nthreads > 0:
            blosc2.set_nthreads(nthreads)
    
    def encode(self, data: np.ndarray) -> bytes:
        """Encode numpy array to compressed bytes."""
        return blosc2.compress2(
            data,
            cname=self.cname,
            clevel=self.clevel,
            shuffle=self.shuffle,
            blocksize=self.blocksize
        )
    
    def decode(self, data: bytes) -> np.ndarray:
        """Decode compressed bytes to numpy array."""
        return blosc2.decompress2(data)
    
    def compress(self, data: np.ndarray) -> bytes:
        """Alias for encode."""
        return self.encode(data)
    
    def decompress(self, data: bytes) -> bytes:
        """Decompress to raw bytes."""
        return blosc2.decompress2(data)


class LZ4Compressor:
    """Standalone LZ4 compressor wrapper."""
    
    def __init__(self, compression_level: int = 3):
        if not LZ4_AVAILABLE:
            raise ImportError("LZ4 not available")
        
        self.compression_level = min(compression_level, 16)  # LZ4 max level
    
    def encode(self, data: np.ndarray) -> bytes:
        """Encode numpy array to compressed bytes."""
        return lz4.frame.compress(
            data.tobytes(),
            compression_level=self.compression_level,
            auto_flush=True
        )
    
    def decode(self, data: bytes) -> bytes:
        """Decode compressed bytes."""
        return lz4.frame.decompress(data)
    
    def compress(self, data: np.ndarray) -> bytes:
        """Alias for encode."""
        return self.encode(data)
    
    def decompress(self, data: bytes) -> bytes:
        """Alias for decode."""
        return self.decode(data)


def benchmark_compressors(data: np.ndarray, 
                         compressors: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Benchmark different compressors on sample data."""
    results = {}
    
    for name, compressor in compressors.items():
        if compressor is None:
            continue
        
        try:
            # Compression benchmark
            start_time = time.time()
            compressed = compressor.encode(data)
            compression_time = time.time() - start_time
            
            # Decompression benchmark
            start_time = time.time()
            decompressed = compressor.decode(compressed)
            decompression_time = time.time() - start_time
            
            # Calculate metrics
            compression_ratio = len(data.tobytes()) / len(compressed)
            compression_speed = (data.nbytes / (1024 * 1024)) / compression_time
            decompression_speed = (data.nbytes / (1024 * 1024)) / decompression_time
            
            results[name] = {
                "compression_ratio": compression_ratio,
                "compression_speed_mb_s": compression_speed,
                "decompression_speed_mb_s": decompression_speed,
                "compression_time": compression_time,
                "decompression_time": decompression_time
            }
            
        except Exception as e:
            results[name] = {"error": str(e)}
    
    return results
