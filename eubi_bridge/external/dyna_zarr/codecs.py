"""
Unified compression configuration for Zarr v2 and v3.

Handles the differences between:
- Zarr v2: Uses numcodecs compressor objects
- Zarr v3: Uses codec pipeline (list of codec configurations)
"""


class Codecs:
    """
    Unified compression configuration for Zarr v2 and v3.
    
    Handles the differences between:
    - Zarr v2: Uses numcodecs compressor objects
    - Zarr v3: Uses codec pipeline (list of codec configurations)
    
    Examples:
        # Blosc compression with LZ4
        codecs = Codecs('blosc', clevel=5, cname='lz4')
        
        # ZSTD compression
        codecs = Codecs('zstd', clevel=3)
        
        # No compression
        codecs = Codecs(None)
    """
    
    def __init__(self, compressor='blosc', clevel=5, cname='lz4', shuffle=1):
        """
        Initialize compression configuration.
        
        Args:
            compressor: Compression algorithm ('blosc', 'zstd', 'gzip', 'lz4', 'bz2', None)
            clevel: Compression level (1-9, higher = more compression but slower)
            cname: Blosc compressor name ('lz4', 'zstd', 'zlib', 'snappy', 'blosclz')
            shuffle: Blosc shuffle mode (0=no shuffle, 1=byte shuffle, 2=bit shuffle)
        """
        self.compressor = compressor
        self.clevel = clevel
        self.cname = cname
        self.shuffle = shuffle
    
    def to_v2_config(self):
        """
        Generate compressor configuration dict for Zarr v2 (TensorStore-compatible).
        
        Returns:
            Dictionary with compressor configuration or None
        """
        if self.compressor == 'blosc':
            return {
                'id': 'blosc',
                'cname': self.cname,
                'clevel': self.clevel,
                'shuffle': self.shuffle
            }
        elif self.compressor == 'zstd':
            return {
                'id': 'zstd',
                'level': self.clevel
            }
        elif self.compressor == 'gzip':
            return {
                'id': 'gzip',
                'level': self.clevel
            }
        elif self.compressor == 'lz4':
            return {
                'id': 'lz4'
            }
        elif self.compressor == 'bz2':
            return {
                'id': 'bz2',
                'level': self.clevel
            }
        elif self.compressor is None or self.compressor == 'none':
            return None
        else:
            raise ValueError(f"Unsupported compressor: {self.compressor}")
    
    def to_numcodecs(self):
        """
        Generate numcodecs compressor object for Zarr v2.
        
        Returns:
            numcodecs compressor object or None
        """
        if self.compressor == 'blosc':
            import numcodecs
            return numcodecs.Blosc(cname=self.cname, clevel=self.clevel, shuffle=self.shuffle)
        elif self.compressor == 'zstd':
            import numcodecs
            return numcodecs.Zstd(level=self.clevel)
        elif self.compressor == 'gzip':
            import numcodecs
            return numcodecs.GZip(level=self.clevel)
        elif self.compressor == 'lz4':
            import numcodecs
            return numcodecs.LZ4()
        elif self.compressor == 'bz2':
            import numcodecs
            return numcodecs.BZ2(level=self.clevel)
        elif self.compressor is None or self.compressor == 'none':
            return None
        else:
            raise ValueError(f"Unsupported compressor: {self.compressor}")
    
    def to_v3_config(self):
        """
        Generate codec pipeline for Zarr v3.
        
        Returns:
            List of codec configurations
        """
        codecs = [
            {'name': 'bytes', 'configuration': {'endian': 'little'}}
        ]
        
        if self.compressor == 'blosc':
            # Convert shuffle integer to string for TensorStore
            shuffle_map = {0: 'noshuffle', 1: 'shuffle', 2: 'bitshuffle'}
            shuffle_str = shuffle_map.get(self.shuffle, 'shuffle')
            
            codecs.append({
                'name': 'blosc',
                'configuration': {
                    'cname': self.cname,
                    'clevel': self.clevel,
                    'shuffle': shuffle_str
                }
            })
        elif self.compressor == 'zstd':
            codecs.append({
                'name': 'zstd',
                'configuration': {'level': self.clevel}
            })
        elif self.compressor == 'gzip':
            codecs.append({
                'name': 'gzip',
                'configuration': {'level': self.clevel}
            })
        elif self.compressor == 'lz4':
            codecs.append({
                'name': 'lz4',
                'configuration': {}
            })
        elif self.compressor == 'bz2':
            codecs.append({
                'name': 'bz2',
                'configuration': {'level': self.clevel}
            })
        elif self.compressor is None or self.compressor == 'none':
            pass  # No compression codec
        else:
            raise ValueError(f"Unsupported compressor: {self.compressor}")
        
        return codecs
    
    @classmethod
    def from_numcodecs(cls, compressor):
        """
        Create Codecs from a numcodecs compressor object.
        
        Args:
            compressor: numcodecs compressor object
            
        Returns:
            Codecs instance
        """
        if compressor is None:
            return cls(compressor=None)
        
        compressor_type = type(compressor).__name__.lower()
        
        if 'blosc' in compressor_type:
            return cls(
                compressor='blosc',
                clevel=getattr(compressor, 'clevel', 5),
                cname=getattr(compressor, 'cname', 'lz4'),
                shuffle=getattr(compressor, 'shuffle', 1)
            )
        elif 'zstd' in compressor_type:
            return cls(
                compressor='zstd',
                clevel=getattr(compressor, 'level', 5)
            )
        elif 'gzip' in compressor_type:
            return cls(
                compressor='gzip',
                clevel=getattr(compressor, 'level', 5)
            )
        elif 'lz4' in compressor_type:
            return cls(compressor='lz4')
        elif 'bz2' in compressor_type:
            return cls(
                compressor='bz2',
                clevel=getattr(compressor, 'level', 5)
            )
        else:
            raise ValueError(f"Unsupported numcodecs compressor: {type(compressor)}")
