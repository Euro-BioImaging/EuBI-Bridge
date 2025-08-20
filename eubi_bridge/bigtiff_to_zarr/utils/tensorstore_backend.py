"""
TensorStore Backend for BigTIFF to OME-NGFF Conversion
Provides alternative backend using tensorstore for data saving and downscaling operations.
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

try:
    import tensorstore as ts
    TENSORSTORE_AVAILABLE = True
except ImportError:
    TENSORSTORE_AVAILABLE = False
    ts = None
    print("Warning: tensorstore not available. TensorStore backend disabled.")

from rich.console import Console

console = Console()


class TensorStoreBackend:
    """TensorStore backend for OME-NGFF data operations."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_path = Path(config.get('output_path'))
        self.zarr_format = config.get('zarr_format', 2)

        if not TENSORSTORE_AVAILABLE:
            raise ImportError("tensorstore is required for TensorStore backend but not available")

    async def create_zarr_array(self, 
                                shape: Tuple[int, ...], 
                                dtype: np.dtype,
                                chunks: Tuple[int, ...], 
                                level: int = 0) -> ts.TensorStore:
        """Create a new zarr array using tensorstore."""

        # Determine zarr array path
        array_path = self.output_path / str(level)
        array_path.mkdir(parents=True, exist_ok=True)

        # Configure tensorstore spec
        spec = {
            'driver': 'zarr',
            'kvstore': {
                'driver': 'file',
                'path': str(array_path)
            },
            'metadata': {
                'chunks': list(chunks),
                'dtype': self._convert_dtype_for_zarr(dtype),
                'shape': list(shape),
                'order': 'C',
                'dimension_separator': '/',
                'zarr_format': 2,
                'fill_value': 0,
                'compressor': self._get_compressor_config()
            }
        }

        console.print(f"[blue]📊 Creating TensorStore array: {array_path}[/blue]")
        console.print(f"[blue]   Shape: {shape}, Dtype: {dtype}, Chunks: {chunks}[/blue]")

        # Create the tensorstore
        store = await ts.open(spec, create=True, delete_existing=True)

        return store

    def _convert_dtype_for_zarr(self, dtype: np.dtype) -> str:
        """Convert numpy dtype to zarr-compatible string format."""
        if dtype == np.uint8:
            return '|u1'
        elif dtype == np.uint16:
            return '<u2'
        elif dtype == np.uint32:
            return '<u4'
        elif dtype == np.int8:
            return '|i1'
        elif dtype == np.int16:
            return '<i2'
        elif dtype == np.int32:
            return '<i4'
        elif dtype == np.float32:
            return '<f4'
        elif dtype == np.float64:
            return '<f8'
        else:
            return str(dtype)

    def _get_compressor_config(self) -> Optional[Dict[str, Any]]:
        """Get compressor configuration for zarr."""
        compression = self.config.get('compression', 'blosc2-lz4')
        if compression == 'none':
            return None

        if compression.startswith('blosc'):
            return {
                'id': 'blosc',
                'cname': 'lz4' if 'lz4' in compression else 'zstd',
                'clevel': self.config.get('compression_level', 3),
                'shuffle': 1
            }
        elif compression == 'lz4':
            return {
                'id': 'lz4',
                'acceleration': 1
            }
        return None

    async def write_chunk_data(self, store: ts.TensorStore, data: np.ndarray,
                             chunk_indices: Tuple[slice, ...]) -> None:
        """Write chunk data to tensorstore array."""
        try:
            # Write data using tensorstore indexing
            await store[chunk_indices].write(data)
        except Exception as e:
            console.print(f"[red]❌ TensorStore write error: {e}[/red]")
            raise

    async def create_downscaled_level(self, ### TODO: update this and simply create output store from the downscaled virtual view
                                      source_store: ts.TensorStore,
                                      target_shape: Tuple[int, ...],
                                      target_chunks: Tuple[int, ...],
                                      scale_factors: Dict[str, int],
                                      axes: str, 
                                      level: int
                                      ) -> ts.TensorStore:
        """Create downscaled pyramid level using tensorstore built-in operations."""
        print(f"🔽 Creating TensorStore downscaled level with scale factors {scale_factors}")
        console.print(f"[blue]🔽 Creating TensorStore downscaled level {level}[/blue]")
        # console.print(f"[blue]   Target shape: {target_shape}[/blue]")
        console.print(f"[blue]   Scale factors: {scale_factors}[/blue]")

        # Create target array specification directly
        level_path = self.output_path / str(level)
        level_path.mkdir(parents=True, exist_ok=True)
        
        scale_factors_list = [scale_factors.get(axis) for axis in axes if axis in scale_factors]

        spec = {
            'driver': 'zarr',
            'kvstore': {
                'driver': 'file',
                'path': str(level_path)
            },
            'metadata': {
                'shape': list(target_shape),
                'chunks': list(target_chunks),
                'dtype': self._convert_dtype_for_zarr(source_store.dtype),
                'dimension_separator': '/',
                'compressor': self._get_compressor_config(),
                'order': 'C',
                'zarr_format': 2,
                'fill_value': 0
            }
        }

        target_store = await ts.open(spec, create=True, open=True)
        # Use tensorstore's built-in downscaling with mean reduction
        await self._downsample_with_tensorstore(
            source_store,
            target_store,
            scale_factors,
            axes
        )

        return target_store

    async def _downsample_with_tensorstore(self, 
                                           source_store: ts.TensorStore,
                                           target_store: ts.TensorStore,
                                           scale_factors: Dict[str, int],
                                           axes: str) -> None:
        """Downsample using tensorstore's built-in operations."""

        console.print(f"[blue]📊 Using TensorStore built-in downscaling operations[/blue]")
        print(f"axes: {axes}")
        # Build downsampling transform using tensorstore
        # Create scale transform for each axis
        scale_transform = []
        print(f"scale_factors: {scale_factors}")
        for i, axis in enumerate(axes):
            if axis in scale_factors:
                scale_factor = scale_factors.get(axis)
                scale_transform.append(scale_factor)

        try:
            # Method 1: Use tensorstore virtual views with downsampling
            console.print(f"[blue]   Using TensorStore virtual downsampling[/blue]")

            # Create virtual downsampled view using tensorstore indexing operations
            # This is more efficient as it uses tensorstore's internal optimizations
            print(f"scale_transform: {scale_transform}")
            downsampled_view = self._create_downsampled_view(source_store, scale_transform)
            print(f"downsampled_view: {downsampled_view}")
            if downsampled_view is not None:
                console.print(f"[blue]   Virtual view created successfully[/blue]")
                # Copy data from virtual view to target store efficiently
                await self._copy_virtual_to_target(downsampled_view, target_store)
            else:
                # Method 2: Use strided indexing for downsampling
                console.print(f"[blue]   Using strided indexing for downsampling[/blue]")
                # Create strided slice for each dimension
                slice_tuple = tuple(
                    slice(None, None, scale_transform[i])
                    for i in range(len(scale_transform))
                )

                # Read downsampled data using strided indexing
                downsampled_data = await source_store[slice_tuple].read()

                # Write to target store
                await target_store[:].write(downsampled_data)

        except Exception as e:
            console.print(f"[yellow]⚠️ TensorStore downsampling failed, using fallback: {e}[/yellow]")
            # Fallback to manual chunked processing
            await self._manual_downsample_fallback(
                source_store, target_store, scale_factors, axes
            )

    def _get_target_chunk_sizes(self, shape: Tuple[int, ...], axes: str) -> Tuple[int, ...]:
        """Get target chunk sizes based on configuration."""
        chunk_sizes = self.config.get('chunk_sizes', {})
        target_chunks = []

        for i, axis in enumerate(axes):
            axis_name = self._axis_to_name(axis)
            chunk_size = chunk_sizes.get(axis_name, 96)  # Default chunk size
            target_chunks.append(min(chunk_size, shape[i]))

        return tuple(target_chunks)

    def _axis_to_name(self, axis: str) -> str:
        """Convert axis letter to dimension name."""
        mapping = {'t': 'time', 'c': 'channel', 'z': 'z', 'y': 'y', 'x': 'x'}
        return mapping.get(axis, axis)

    def _generate_chunk_indices(self, shape: Tuple[int, ...],
                              chunks: Tuple[int, ...]) -> List[Tuple[slice, ...]]:
        """Generate all chunk indices for the given shape and chunk sizes."""
        indices_list = []

        def _generate_recursive(dim: int, current_indices: List[slice]):
            if dim == len(shape):
                indices_list.append(tuple(current_indices))
                return

            for start in range(0, shape[dim], chunks[dim]):
                end = min(start + chunks[dim], shape[dim])
                current_indices.append(slice(start, end))
                _generate_recursive(dim + 1, current_indices)
                current_indices.pop()

        _generate_recursive(0, [])
        return indices_list

    def _calculate_source_indices(self, target_indices: Tuple[slice, ...],
                                scale_factors: Dict[str, int],
                                axes: str) -> Tuple[slice, ...]:
        """Calculate source indices from target indices and scale factors."""
        source_indices = []

        for i, target_slice in enumerate(target_indices):
            axis = axes[i]
            axis_name = self._axis_to_name(axis)
            scale_factor = scale_factors.get(axis_name, 1)

            if scale_factor > 1:
                # Scale up the indices
                start = target_slice.start * scale_factor
                stop = target_slice.stop * scale_factor
                source_indices.append(slice(start, stop))
            else:
                # No scaling for this dimension
                source_indices.append(target_slice)

        return tuple(source_indices)

    async def _manual_downsample_fallback(self, source_store: ts.TensorStore,
                                        target_store: ts.TensorStore,
                                        scale_factors: Dict[str, int],
                                        axes: str) -> None:
        """Manual downsampling fallback when tensorstore methods fail."""

        console.print(f"[blue]📊 Using manual downsampling fallback[/blue]")

        source_shape = source_store.shape
        target_shape = target_store.shape

        # Calculate target chunk sizes
        target_chunks = self._get_target_chunk_sizes(target_shape, axes)

        # Process in chunks to avoid memory issues
        total_chunks = np.prod([
            (target_shape[i] + target_chunks[i] - 1) // target_chunks[i]
            for i in range(len(target_shape))
        ])

        console.print(f"[blue]📊 Processing {total_chunks} chunks for manual downscaling[/blue]")

        chunk_count = 0
        for chunk_indices in self._generate_chunk_indices(target_shape, target_chunks):
            # Calculate corresponding source region
            source_indices = self._calculate_source_indices(
                chunk_indices, scale_factors, axes
            )

            # Read source data
            source_data = await source_store[source_indices].read()

            # Perform manual mean pooling downscaling
            downscaled_data = self._downsample_data_manual(
                source_data, scale_factors, axes
            )

            # Write to target
            await target_store[chunk_indices].write(downscaled_data)

            chunk_count += 1
            if chunk_count % 100 == 0:
                console.print(f"[blue]📊 Processed {chunk_count}/{total_chunks} chunks[/blue]")

    def _downsample_data_manual(self, data: np.ndarray, scale_factors: Dict[str, int],
                               axes: str) -> np.ndarray:
        """Manual downsampling using mean pooling as fallback."""

        # Start with original data
        result = data.copy()

        for i, axis in enumerate(axes):
            axis_name = self._axis_to_name(axis)
            scale_factor = scale_factors.get(axis_name, 1)

            if scale_factor > 1:
                # Perform mean pooling along this axis
                axis_size = result.shape[i]
                if axis_size >= scale_factor:
                    # Reshape and take mean
                    new_size = axis_size // scale_factor
                    slices = [slice(None)] * result.ndim
                    slices[i] = slice(0, new_size * scale_factor)

                    # Trim to multiple of scale_factor
                    trimmed = result[tuple(slices)]

                    # Reshape for pooling
                    new_shape = list(trimmed.shape)
                    new_shape[i] = new_size
                    new_shape.insert(i + 1, scale_factor)

                    reshaped = trimmed.reshape(new_shape)

                    # Take mean along the pooling dimension
                    result = np.mean(reshaped, axis=i + 1)

        return result

    def _create_downsampled_view(self,
                                 source_store: ts.TensorStore,
                                 scale_factors: List[int],
                                 method: str = 'stride'
                                 ) -> Optional[ts.TensorStore]:
        """Create a virtual downsampled view using tensorstore operations."""
        print(f"source_store: {source_store}")
        print(f"scale_factors: {scale_factors}")

        # Method 1: Try using tensorstore's virtual downsampling if available
        return ts.downsample(source_store,
                              [int(np.round(factor)) for factor in scale_factors],
                              method=method)

    async def _copy_virtual_to_target(self, 
                                      virtual_view: ts.TensorStore,
                                      target_store: ts.TensorStore
                                      ) -> None:
        """Efficiently copy data from virtual view to target store."""
        try:
            # Use tensorstore's optimized copying if available
            if hasattr(ts, 'copy'):
                await ts.copy(virtual_view, target_store)
            else:
                # Fallback to reading and writing
                data = await virtual_view[:].read()
                await target_store[:].write(data)

        except Exception as e:
            console.print(f"[yellow]⚠️ Virtual copy failed: {e}[/yellow]")
            # Read data and write manually
            data = await virtual_view[:].read()
            await target_store[:].write(data)

    async def save_ome_zarr_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save OME-NGFF metadata using tensorstore-compatible format."""

        # Create .zattrs file for OME-NGFF metadata
        zattrs_path = self.output_path / '.zattrs'

        with open(zattrs_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Create .zgroup file
        zgroup_path = self.output_path / '.zgroup'
        zgroup_data = {'zarr_format': self.zarr_format}

        with open(zgroup_path, 'w') as f:
            json.dump(zgroup_data, f, indent=2)

        console.print(f"[green]✅ TensorStore OME-NGFF metadata saved[/green]")


def create_tensorstore_backend(config: Dict[str, Any]) -> TensorStoreBackend:
    """Factory function to create tensorstore backend."""
    if not TENSORSTORE_AVAILABLE:
        raise ImportError("tensorstore package is required but not installed")

    return TensorStoreBackend(config)