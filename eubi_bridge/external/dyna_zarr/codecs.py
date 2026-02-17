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
        Generate codec pipeline for Zarr v3 using proper zarr.codecs API.
        
        Returns:
            List of codec configurations (from .to_dict() calls)
        """
        from zarr import codecs
        
        codecs_list = [
            codecs.BytesCodec(endian=codecs.Endian.little).to_dict()
        ]
        
        if self.compressor == 'blosc':
            # Use BloscCodec with proper BloscShuffle enum
            shuffle_map = {0: codecs.BloscShuffle.noshuffle, 
                          1: codecs.BloscShuffle.shuffle, 
                          2: codecs.BloscShuffle.bitshuffle}
            shuffle_enum = shuffle_map.get(self.shuffle, codecs.BloscShuffle.shuffle)
            
            blosc_codec = codecs.BloscCodec(
                cname=self.cname,
                clevel=self.clevel,
                shuffle=shuffle_enum
            )
            codecs_list.append(blosc_codec.to_dict())
            
        elif self.compressor == 'zstd':
            zstd_codec = codecs.ZstdCodec(level=self.clevel)
            codecs_list.append(zstd_codec.to_dict())
            
        elif self.compressor == 'gzip':
            gzip_codec = codecs.GzipCodec(level=self.clevel)
            codecs_list.append(gzip_codec.to_dict())
            
        elif self.compressor == 'lz4':
            # LZ4 is not directly available in zarr.codecs v3, would need blosc
            raise ValueError("LZ4 is not available in Zarr v3. Use Blosc with cname='lz4' instead.")
            
        elif self.compressor == 'bz2':
            # BZ2 is not directly available in zarr.codecs v3
            raise ValueError("BZ2 is not available in Zarr v3. Use Zstd or Gzip instead.")
            
        elif self.compressor is None or self.compressor == 'none':
            pass  # No compression codec, only bytes codec
            
        else:
            raise ValueError(f"Unsupported compressor: {self.compressor}")
        
        return codecs_list
    
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
