"""
Array operations and transformations for DynamicArray.

This module provides:
- Transform classes for lazy operations
- operations class with static methods for array manipulations
- Support for all major numpy/dask.array operations
"""

import numpy as np
from typing import Tuple, Union, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from dyna_zarr.dynamic_array import DynamicArray


class Transform:
    """
    Base class for lazy transformations.
    """

    def __init__(self):
        self.shape = None
        self.chunks = None
        self.dtype = None

    def read(self, key):
        raise NotImplementedError


class ConcatenateTransform(Transform):
    """
    Lazy concatenation of multiple arrays along an axis.
    """

    def __init__(self, arrays: Tuple['DynamicArray', ...], axis: int):
        super().__init__()
        self.arrays = arrays
        self.axis = axis

        # Validate shapes
        ref_shape = list(arrays[0].shape)
        for arr in arrays[1:]:
            for i, (s1, s2) in enumerate(zip(ref_shape, arr.shape)):
                if i != axis and s1 != s2:
                    raise ValueError(f"All arrays must have same shape except on axis {axis}")

        # Compute output shape
        self.shape = tuple(
            sum(arr.shape[axis] for arr in arrays) if i == axis else ref_shape[i]
            for i in range(len(ref_shape))
        )

        # Use chunks from first array
        self.chunks = arrays[0].chunks
        self.dtype = arrays[0].dtype

        # Precompute cumulative sizes for fast lookup
        self._cumulative_sizes = [0]
        for arr in arrays:
            self._cumulative_sizes.append(self._cumulative_sizes[-1] + arr.shape[axis])
        
        # OPTIMIZATION: Pre-compute optimal read strategies
        # Store whether arrays can be read sequentially (no interleaving)
        self._sequential_friendly = len(arrays) <= 4

    def read(self, key):
        """
        Read data by routing to appropriate source arrays.
        Optimized with efficient memory allocation and direct writing.
        """
        # Normalize key to tuple of slices
        if not isinstance(key, tuple):
            key = (key,)

        # Pad with full slices if needed
        key = key + (slice(None),) * (len(self.shape) - len(key))

        # Convert to slices
        normalized_key = []
        for k, size in zip(key, self.shape):
            if isinstance(k, int):
                normalized_key.append(slice(k, k + 1))
            elif isinstance(k, slice):
                normalized_key.append(k)
            else:
                raise NotImplementedError(f"Indexing with {type(k)} not supported")

        axis_slice = normalized_key[self.axis]
        start = axis_slice.start if axis_slice.start is not None else 0
        stop = axis_slice.stop if axis_slice.stop is not None else self.shape[self.axis]
        step = axis_slice.step if axis_slice.step is not None else 1

        if step != 1:
            raise NotImplementedError("Step slicing not supported yet")

        # Find which arrays we need to read from
        arrays_to_read = []

        for i, arr in enumerate(self.arrays):
            arr_start = self._cumulative_sizes[i]
            arr_stop = self._cumulative_sizes[i + 1]

            # Check if this array overlaps with requested range
            if start < arr_stop and stop > arr_start:
                # Compute slice within this array
                local_start = max(0, start - arr_start)
                local_stop = min(arr.shape[self.axis], stop - arr_start)

                # Build the key for this array
                local_key = list(normalized_key)
                local_key[self.axis] = slice(local_start, local_stop)
                local_key = tuple(local_key)

                arrays_to_read.append((arr, local_key, local_stop - local_start))

        # OPTIMIZATION: Handle based on number of arrays
        if len(arrays_to_read) == 0:
            raise ValueError("No data to read")
        elif len(arrays_to_read) == 1:
            # Single array - no concatenation needed
            result = arrays_to_read[0][0]._read_direct(arrays_to_read[0][1])
        else:
            # Multiple arrays - concatenate
            # Collect data from all arrays in order
            result_parts = []
            for arr, local_key, _ in arrays_to_read:
                # Use _read_direct to avoid creating more SliceTransforms
                data = arr._read_direct(local_key)
                result_parts.append(data)
            
            result = np.concatenate(result_parts, axis=self.axis)

        # Remove dimensions that were indexed with int
        squeeze_axes = [i for i, k in enumerate(key) if isinstance(k, int)]
        for ax in reversed(squeeze_axes):
            result = np.squeeze(result, axis=ax)

        return result


class StackTransform(Transform):
    """
    Lazy stacking of arrays along a new axis.
    """

    def __init__(self, arrays: Tuple['DynamicArray', ...], axis: int):
        super().__init__()
        self.arrays = arrays
        self.axis = axis

        # Validate all arrays have same shape
        ref_shape = arrays[0].shape
        for arr in arrays[1:]:
            if arr.shape != ref_shape:
                raise ValueError("All arrays must have the same shape for stacking")

        # Compute output shape (insert new dimension)
        self.shape = ref_shape[:axis] + (len(arrays),) + ref_shape[axis:]
        self.chunks = arrays[0].chunks[:axis] + (1,) + arrays[0].chunks[axis:]
        self.dtype = arrays[0].dtype

    def read(self, key):
        # Normalize key
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (len(self.shape) - len(key))

        # Extract index/slice for new axis
        new_axis_key = key[self.axis]

        # Build key for source arrays (remove new axis)
        source_key = key[:self.axis] + key[self.axis + 1:]

        # Determine which arrays to read
        if isinstance(new_axis_key, int):
            # Single array
            return self.arrays[new_axis_key]._read_direct(source_key)
        elif isinstance(new_axis_key, slice):
            start = new_axis_key.start if new_axis_key.start is not None else 0
            stop = new_axis_key.stop if new_axis_key.stop is not None else len(self.arrays)
            step = new_axis_key.step if new_axis_key.step is not None else 1

            parts = [self.arrays[i]._read_direct(source_key) for i in range(start, stop, step)]
            return np.stack(parts, axis=self.axis)


class SliceTransform(Transform):
    """
    Lazy slicing of an array with support for np.newaxis.
    """

    def __init__(self, array: 'DynamicArray', key):
        super().__init__()
        self.array = array
        
        # Normalize key to tuple
        if not isinstance(key, tuple):
            key = (key,)
        
        self.key = key

        # Compute output shape and track new axes
        new_shape = []
        new_chunks = []
        original_dim = 0
        chunks = array.chunks if array.chunks is not None else (1,) * array.ndim

        # Track which original dimensions are being kept
        kept_dims = []

        for k in key:
            if k is np.newaxis:
                # Add new axis with chunk size 1
                new_shape.append(1)
                new_chunks.append(1)
            else:
                if original_dim >= array.ndim:
                    raise IndexError("Too many indices for array")

                if isinstance(k, int):
                    # This dimension will be removed
                    pass
                elif isinstance(k, slice):
                    # Calculate size for this dimension
                    size = array.shape[original_dim]
                    start = k.start if k.start is not None else 0
                    stop = k.stop if k.stop is not None else size
                    step = k.step if k.step is not None else 1
                    new_shape.append((stop - start + step - 1) // step)
                    new_chunks.append(chunks[original_dim])
                    kept_dims.append(original_dim)
                else:
                    raise TypeError(f"Invalid index type: {type(k)}")

                original_dim += 1

        # Add remaining dimensions
        for i in range(original_dim, array.ndim):
            new_shape.append(array.shape[i])
            new_chunks.append(chunks[i])
            kept_dims.append(i)

        self.shape = tuple(new_shape)
        self.chunks = tuple(new_chunks)
        self.dtype = array.dtype
        self.new_axes = [i for i, k in enumerate(key) if k is np.newaxis]

    def read(self, read_key):
        """
        Read data by composing the stored slice with a new read key.
        """
        if not isinstance(read_key, tuple):
            read_key = (read_key,)

        # Pad read_key to match output dimensions
        read_key = read_key + (slice(None),) * (len(self.shape) - len(read_key))

        # Build the key to apply to the underlying array
        full_key = []
        output_dim = 0  # Track dimension in the output (sliced) array
        input_dim = 0  # Track dimension in the input (underlying) array

        for stored_key_elem in self.key:
            if stored_key_elem is np.newaxis:
                # New axis - just track that we're moving to next output dimension
                # but don't consume an input dimension
                output_dim += 1
                continue

            # This is a real dimension in the underlying array
            if isinstance(stored_key_elem, int):
                # Dimension was removed by integer indexing in the stored slice
                full_key.append(stored_key_elem)
                input_dim += 1
                # Don't increment output_dim since this dimension is gone

            elif isinstance(stored_key_elem, slice):
                # Dimension was sliced in the stored slice
                read_elem = read_key[output_dim]

                # Get parameters of the stored slice
                orig_size = self.array.shape[input_dim]
                stored_start = stored_key_elem.start if stored_key_elem.start is not None else 0
                stored_stop = stored_key_elem.stop if stored_key_elem.stop is not None else orig_size
                stored_step = stored_key_elem.step if stored_key_elem.step is not None else 1

                if isinstance(read_elem, (int, np.integer)):
                    # Compose integer index with slice
                    new_idx = stored_start + read_elem * stored_step
                    if new_idx < 0 or new_idx >= orig_size:
                        raise IndexError(f"Index {new_idx} is out of bounds for axis with size {orig_size}")
                    full_key.append(new_idx)

                elif isinstance(read_elem, slice):
                    # Compose two slices
                    read_start = read_elem.start if read_elem.start is not None else 0
                    read_stop = read_elem.stop if read_elem.stop is not None else self.shape[output_dim]
                    read_step = read_elem.step if read_elem.step is not None else 1

                    new_start = stored_start + read_start * stored_step
                    new_stop = stored_start + read_stop * stored_step
                    new_step = stored_step * read_step

                    # Clamp to original array bounds
                    if new_step > 0:
                        new_start = max(0, min(new_start, orig_size))
                        new_stop = max(0, min(new_stop, orig_size))
                    else:
                        new_start = min(orig_size - 1, max(-1, new_start))
                        new_stop = min(orig_size - 1, max(-1, new_stop))

                    full_key.append(slice(new_start, new_stop, new_step))
                else:
                    raise TypeError(f"Invalid index type: {type(read_elem)}")

                input_dim += 1
                output_dim += 1
            else:
                raise TypeError(f"Invalid stored key type: {type(stored_key_elem)}")

        # Add any remaining dimensions that weren't in the stored key
        while input_dim < self.array.ndim:
            if output_dim < len(read_key):
                full_key.append(read_key[output_dim])
                output_dim += 1
            else:
                full_key.append(slice(None))
            input_dim += 1

        # Read from underlying array
        result = self.array._read_direct(tuple(full_key))

        # Add back any newaxis dimensions at the correct positions
        for i, stored_key_elem in enumerate(self.key):
            if stored_key_elem is np.newaxis:
                # Count how many non-newaxis, non-int dimensions come before this
                axis_pos = sum(1 for j, k in enumerate(self.key[:i])
                               if k is not np.newaxis and not isinstance(k, int))
                result = np.expand_dims(result, axis=axis_pos)

        return result


class ExpandDimsTransform(Transform):
    """
    Lazy addition of a singleton dimension to an array.
    """

    def __init__(self, array: 'DynamicArray', axis: int):
        super().__init__()
        self.array = array
        self.axis = axis if axis >= 0 else array.ndim + axis + 1

        # Compute new shape and chunks
        self.shape = array.shape[:self.axis] + (1,) + array.shape[self.axis:]

        # Handle chunks - if chunks is None, keep it as None
        if array.chunks is not None:
            self.chunks = array.chunks[:self.axis] + (1,) + array.chunks[self.axis:]
        else:
            self.chunks = None

        self.dtype = array.dtype

    def read(self, key):
        """Read with the singleton dimension inserted at the correct axis."""
        # Normalize key to tuple
        if not isinstance(key, tuple):
            key = (key,)
        
        # Pad key with full slices if needed
        key = key + (slice(None),) * (len(self.shape) - len(key))
        
        # Build the key for the underlying array by removing the element at self.axis
        # since the underlying array doesn't have this dimension yet
        underlying_key = key[:self.axis] + key[self.axis + 1:]
        
        # Read from underlying array
        result = self.array._read_direct(underlying_key)
        
        # Add back the singleton dimension at the correct axis
        result = np.expand_dims(result, axis=self.axis)
        
        return result


class SwapAxesTransform(Transform):
    """
    Lazy swapping of two axes in an array.
    """

    def __init__(self, array: 'DynamicArray', axis1: int, axis2: int):
        super().__init__()
        self.array = array
        self.axis1 = axis1 if axis1 >= 0 else array.ndim + axis1
        self.axis2 = axis2 if axis2 >= 0 else array.ndim + axis2

        # Validate axes
        if self.axis1 >= array.ndim or self.axis2 >= array.ndim:
            raise ValueError(f"Axis out of bounds for {array.ndim}-D array")

        # Compute new shape and chunks
        shape = list(array.shape)
        chunks = list(array.chunks)
        shape[self.axis1], shape[self.axis2] = shape[self.axis2], shape[self.axis1]
        chunks[self.axis1], chunks[self.axis2] = chunks[self.axis2], chunks[self.axis1]

        self.shape = tuple(shape)
        self.chunks = tuple(chunks)
        self.dtype = array.dtype

    def read(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        # Pad key with full slices if needed
        key = key + (slice(None),) * (len(self.shape) - len(key))
        key = list(key)
        
        # Swap the key indices to match the original array's axes
        original_key = key[:]
        original_key[self.axis1], original_key[self.axis2] = original_key[self.axis2], original_key[self.axis1]

        # Read from underlying array with unswapped key
        result = self.array._read_direct(tuple(original_key))
        
        # Swap the axes in the result back
        result = np.swapaxes(result, self.axis1, self.axis2)
        
        return result


class TransposeTransform(Transform):
    """
    Lazy transposition (reordering) of array axes.
    """

    def __init__(self, array: 'DynamicArray', axes: tuple):
        super().__init__()
        self.array = array
        self.axes = tuple(ax if ax >= 0 else array.ndim + ax for ax in axes)

        # Validate axes
        if len(self.axes) != array.ndim:
            raise ValueError(f"axes don't match array (expected {array.ndim} dimensions, got {len(axes)})")
        if len(set(self.axes)) != len(self.axes):
            raise ValueError("repeated axis in transpose")
        if any(ax >= array.ndim or ax < 0 for ax in self.axes):
            raise ValueError("axis out of bounds for array")

        # Compute new shape and chunks
        self.shape = tuple(array.shape[ax] for ax in self.axes)
        if array.chunks is not None:
            self.chunks = tuple(array.chunks[ax] for ax in self.axes)
        else:
            self.chunks = None
        self.dtype = array.dtype

    def read(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        # Pad key with full slices if needed
        key = key + (slice(None),) * (len(self.shape) - len(key))

        # Compute inverse permutation to map from transposed axes back to original
        inv_axes = [0] * len(self.axes)
        for i, ax in enumerate(self.axes):
            inv_axes[ax] = i

        # Reorder the key according to the inverse permutation
        reordered_key = tuple(key[inv_axes[i]] for i in range(len(self.axes)))

        # Read from underlying array with reordered key
        result = self.array._read_direct(reordered_key)

        # Now transpose the result to match the expected output order
        result = np.transpose(result, self.axes)

        return result


# Extended operations from extended_operations.py

class ReshapeTransform(Transform):
    """Lazy reshape without data copying."""
    
    def __init__(self, array: 'DynamicArray', shape: Tuple[int, ...]):
        super().__init__()
        self.array = array
        self.new_shape = shape
        
        # Validate reshape is possible
        if np.prod(array.shape) != np.prod(shape):
            raise ValueError(f"Cannot reshape array of size {np.prod(array.shape)} into shape {shape}")
        
        self.shape = tuple(shape)
        self.chunks = None  # Reshaping invalidates chunk alignment
        self.dtype = array.dtype
    
    def read(self, key):
        # For reshape, we need to materialize and reshape
        data = self.array._read_direct(slice(None))
        return data.reshape(self.new_shape)[key]


class SqueezeTransform(Transform):
    """Remove singleton dimensions.
    
    Supports NumPy-compatible squeeze with axis as None, int, or tuple of ints.
    """
    
    def __init__(self, array: 'DynamicArray', axis: Union[int, Tuple[int, ...], None] = None):
        super().__init__()
        self.array = array
        self.axis = axis
        
        # Normalize axis to tuple of indices to squeeze
        if axis is None:
            # Remove all singleton dimensions
            self.squeeze_axes = [i for i, s in enumerate(array.shape) if s == 1]
        elif isinstance(axis, int):
            # Single axis
            axis = axis if axis >= 0 else array.ndim + axis
            if axis < 0 or axis >= array.ndim:
                raise ValueError(f"axis {axis} is out of bounds for array of dimension {array.ndim}")
            if array.shape[axis] != 1:
                raise ValueError(f"Cannot squeeze axis {axis} with size {array.shape[axis]}")
            self.squeeze_axes = [axis]
        elif isinstance(axis, (tuple, list)):
            # Multiple axes
            squeeze_axes = []
            for ax in axis:
                ax = ax if ax >= 0 else array.ndim + ax
                if ax < 0 or ax >= array.ndim:
                    raise ValueError(f"axis {ax} is out of bounds for array of dimension {array.ndim}")
                if array.shape[ax] != 1:
                    raise ValueError(f"Cannot squeeze axis {ax} with size {array.shape[ax]}")
                squeeze_axes.append(ax)
            self.squeeze_axes = sorted(set(squeeze_axes))  # Remove duplicates and sort
        else:
            raise TypeError(f"axis must be None, int, or tuple of ints, got {type(axis)}")
        
        # Compute output shape
        self.shape = tuple(s for i, s in enumerate(array.shape) if i not in self.squeeze_axes)
        
        # Handle chunks
        if array.chunks:
            new_chunks = [c for i, c in enumerate(array.chunks) if i not in self.squeeze_axes]
            self.chunks = tuple(new_chunks) if new_chunks else (1,)
        else:
            self.chunks = None
        
        self.dtype = array.dtype
    
    def read(self, key):
        """
        Lazy read that only accesses the required region of the underlying array.
        """
        # Normalize key to tuple
        if not isinstance(key, tuple):
            key = (key,)
        
        # Build unsqueezed key by inserting slice(None) for squeezed axes
        # Track which positions in unsqueezed_key correspond to squeezed axes
        unsqueezed_key = []
        axes_to_squeeze_in_result = []
        output_idx = 0
        for input_idx in range(self.array.ndim):
            if input_idx in self.squeeze_axes:
                # This axis was squeezed - insert full slice
                unsqueezed_key.append(slice(None))
                # Mark this position for squeezing in the result
                axes_to_squeeze_in_result.append(len(unsqueezed_key) - 1)
            else:
                # This axis is preserved - use key from read operation
                if output_idx < len(key):
                    unsqueezed_key.append(key[output_idx])
                else:
                    unsqueezed_key.append(slice(None))
                output_idx += 1
        
        # Read only the required region from underlying array
        result = self.array._read_direct(tuple(unsqueezed_key))
        
        
        # Squeeze the marked axes (in reverse order to avoid index shifting)
        for ax in sorted(axes_to_squeeze_in_result, reverse=True):
            if ax < len(result.shape) and result.shape[ax] == 1:
                result = np.squeeze(result, axis=ax)
        
        
        return result


class FlattenTransform(Transform):
    """Flatten array to 1D."""
    
    def __init__(self, array: 'DynamicArray'):
        super().__init__()
        self.array = array
        self.original_shape = array.shape
        self.shape = (np.prod(array.shape),)
        self.chunks = None
        self.dtype = array.dtype
    
    def read(self, key):
        return self.array._read_direct(slice(None)).flatten()[key]


class PadTransform(Transform):
    """Lazy padding (materializes on read)."""
    
    def __init__(self, array: 'DynamicArray', pad_width: Union[int, Tuple]):
        super().__init__()
        self.array = array
        
        # Normalize pad_width
        if isinstance(pad_width, int):
            self.pad_width = [(pad_width, pad_width)] * array.ndim
        else:
            self.pad_width = pad_width
        
        # Calculate output shape
        new_shape = []
        for s, (pad_before, pad_after) in zip(array.shape, self.pad_width):
            new_shape.append(s + pad_before + pad_after)
        
        self.shape = tuple(new_shape)
        self.chunks = None
        self.dtype = array.dtype
    
    def read(self, key):
        data = self.array._read_direct(slice(None))
        return np.pad(data, self.pad_width)[key]


class TileTransform(Transform):
    """Repeat array along dimensions."""
    
    def __init__(self, array: 'DynamicArray', reps: Union[int, Tuple]):
        super().__init__()
        self.array = array
        
        if isinstance(reps, int):
            reps = (reps,) * array.ndim
        self.reps = reps
        
        # Calculate output shape
        self.shape = tuple(s * r for s, r in zip(array.shape, reps))
        self.chunks = None
        self.dtype = array.dtype
    
    def read(self, key):
        data = self.array._read_direct(slice(None))
        return np.tile(data, self.reps)[key]


class RollTransform(Transform):
    """Roll array elements along an axis."""
    
    def __init__(self, array: 'DynamicArray', shift: int, axis: Optional[int] = None):
        super().__init__()
        self.array = array
        self.shift = shift
        self.axis = axis
        self.shape = array.shape
        self.chunks = array.chunks
        self.dtype = array.dtype
    
    def read(self, key):
        data = self.array._read_direct(slice(None))
        return np.roll(data, self.shift, axis=self.axis)[key]


class FlipTransform(Transform):
    """Flip/reverse array along an axis."""
    
    def __init__(self, array: 'DynamicArray', axis: int):
        super().__init__()
        self.array = array
        self.axis = axis
        self.shape = array.shape
        self.chunks = array.chunks
        self.dtype = array.dtype
    
    def read(self, key):
        data = self.array._read_direct(slice(None))
        return np.flip(data, axis=self.axis)[key]


class ClipTransform(Transform):
    """Clip array values to a range."""
    
    def __init__(self, array: 'DynamicArray', a_min: Optional[float], a_max: Optional[float]):
        super().__init__()
        self.array = array
        self.a_min = a_min
        self.a_max = a_max
        self.shape = array.shape
        self.chunks = array.chunks
        self.dtype = array.dtype
    
    def read(self, key):
        data = self.array._read_direct(key)
        return np.clip(data, self.a_min, self.a_max)


class AbsTransform(Transform):
    """Absolute value."""
    
    def __init__(self, array: 'DynamicArray'):
        super().__init__()
        self.array = array
        self.shape = array.shape
        self.chunks = array.chunks
        self.dtype = array.dtype
    
    def read(self, key):
        return np.abs(self.array._read_direct(key))


class SignTransform(Transform):
    """Sign of array elements."""
    
    def __init__(self, array: 'DynamicArray'):
        super().__init__()
        self.array = array
        self.shape = array.shape
        self.chunks = array.chunks
        self.dtype = array.dtype
    
    def read(self, key):
        return np.sign(self.array._read_direct(key))


class RoundTransform(Transform):
    """Round array elements."""
    
    def __init__(self, array: 'DynamicArray', decimals: int = 0):
        super().__init__()
        self.array = array
        self.decimals = decimals
        self.shape = array.shape
        self.chunks = array.chunks
        self.dtype = array.dtype
    
    def read(self, key):
        return np.round(self.array._read_direct(key), decimals=self.decimals)


class SqrtTransform(Transform):
    """Square root."""
    
    def __init__(self, array: 'DynamicArray'):
        super().__init__()
        self.array = array
        self.shape = array.shape
        self.chunks = array.chunks
        self.dtype = np.float64
    
    def read(self, key):
        return np.sqrt(self.array._read_direct(key))


class WhereTransform(Transform):
    """Conditional element selection."""
    
    def __init__(self, condition: 'DynamicArray', x: 'DynamicArray', y: 'DynamicArray'):
        super().__init__()
        if not (condition.shape == x.shape == y.shape):
            raise ValueError("All arrays must have the same shape")
        
        self.condition = condition
        self.x = x
        self.y = y
        self.shape = x.shape
        self.chunks = x.chunks
        self.dtype = x.dtype
    
    def read(self, key):
        cond = self.condition._read_direct(key)
        x_data = self.x._read_direct(key)
        y_data = self.y._read_direct(key)
        return np.where(cond, x_data, y_data)


class MultiplyTransform(Transform):
    """Element-wise multiplication."""
    
    def __init__(self, array1: 'DynamicArray', array2: Union['DynamicArray', float]):
        super().__init__()
        self.array1 = array1
        self.array2 = array2
        self.shape = array1.shape
        self.chunks = array1.chunks
        self.dtype = array1.dtype
    
    def read(self, key):
        # Import here to avoid circular dependency
        from dyna_zarr.dynamic_array import DynamicArray
        
        data1 = self.array1._read_direct(key)
        if isinstance(self.array2, DynamicArray):
            data2 = self.array2._read_direct(key)
        else:
            data2 = self.array2
        return data1 * data2


class AddTransform(Transform):
    """Element-wise addition."""
    
    def __init__(self, array1: 'DynamicArray', array2: Union['DynamicArray', float]):
        super().__init__()
        self.array1 = array1
        self.array2 = array2
        self.shape = array1.shape
        self.chunks = array1.chunks
        self.dtype = array1.dtype
    
    def read(self, key):
        # Import here to avoid circular dependency
        from dyna_zarr.dynamic_array import DynamicArray
        
        data1 = self.array1._read_direct(key)
        if isinstance(self.array2, DynamicArray):
            data2 = self.array2._read_direct(key)
        else:
            data2 = self.array2
        return data1 + data2


class MinTransform(Transform):
    """Lazy minimum reduction along specified axes."""
    def __init__(self, array: 'DynamicArray', axis: Optional[int] = None):
        super().__init__()
        self.array = array
        self.axis = axis
        
        if axis is None:
            self.shape = ()
            self.chunks = None
        else:
            normalized_axis = axis if axis >= 0 else array.ndim + axis
            if normalized_axis < 0 or normalized_axis >= array.ndim:
                raise ValueError(f"axis {axis} out of bounds for dimension {array.ndim}")
            self.shape = array.shape[:normalized_axis] + array.shape[normalized_axis + 1:]
            self.chunks = array.chunks[:normalized_axis] + array.chunks[normalized_axis + 1:] if array.chunks else None
        
        self.dtype = array.dtype
    
    def read(self, key):
        """Read entire array and compute minimum."""
        full_data = self.array._read_direct(tuple(slice(None) for _ in range(self.array.ndim)))
        return np.min(full_data, axis=self.axis)


class MaxTransform(Transform):
    """Lazy maximum reduction along specified axes."""
    def __init__(self, array: 'DynamicArray', axis: Optional[int] = None):
        super().__init__()
        self.array = array
        self.axis = axis
        
        if axis is None:
            self.shape = ()
            self.chunks = None
        else:
            normalized_axis = axis if axis >= 0 else array.ndim + axis
            if normalized_axis < 0 or normalized_axis >= array.ndim:
                raise ValueError(f"axis {axis} out of bounds for dimension {array.ndim}")
            self.shape = array.shape[:normalized_axis] + array.shape[normalized_axis + 1:]
            self.chunks = array.chunks[:normalized_axis] + array.chunks[normalized_axis + 1:] if array.chunks else None
        
        self.dtype = array.dtype
    
    def read(self, key):
        """Read entire array and compute maximum."""
        full_data = self.array._read_direct(tuple(slice(None) for _ in range(self.array.ndim)))
        return np.max(full_data, axis=self.axis)


# operations class with static methods for creating transforms

class operations:
    """
    Array operations following numpy API conventions.
    All methods return lazy DynamicArray objects with transforms applied.
    """
    
    @staticmethod
    def expand_dims(array: 'DynamicArray', axis: int) -> 'DynamicArray':
        """Add a new axis of length 1."""
        transform = ExpandDimsTransform(array, axis)
        return array._with_transform(transform)
    
    @staticmethod
    def concatenate(arrays: List['DynamicArray'], axis: int = 0) -> 'DynamicArray':
        """Concatenate arrays along an existing axis."""
        if not arrays:
            raise ValueError("Need at least one array to concatenate")
        transform = ConcatenateTransform(tuple(arrays), axis)
        return arrays[0]._with_transform(transform)
    
    @staticmethod
    def stack(arrays: List['DynamicArray'], axis: int = 0) -> 'DynamicArray':
        """Stack arrays along a new axis."""
        if not arrays:
            raise ValueError("Need at least one array to stack")
        transform = StackTransform(tuple(arrays), axis)
        return arrays[0]._with_transform(transform)
    
    @staticmethod
    def swap_axes(array: 'DynamicArray', axis1: int, axis2: int) -> 'DynamicArray':
        """Swap two axes."""
        transform = SwapAxesTransform(array, axis1, axis2)
        return array._with_transform(transform)
    
    @staticmethod
    def transpose(array: 'DynamicArray', axes: Tuple[int, ...]) -> 'DynamicArray':
        """Permute array dimensions."""
        transform = TransposeTransform(array, axes)
        return array._with_transform(transform)
    
    @staticmethod
    def reshape(array: 'DynamicArray', shape: Tuple[int, ...]) -> 'DynamicArray':
        """Reshape array to new shape."""
        transform = ReshapeTransform(array, shape)
        return array._with_transform(transform)
    
    @staticmethod
    def squeeze(array: 'DynamicArray', axis: Optional[int] = None) -> 'DynamicArray':
        """Remove singleton dimensions."""
        transform = SqueezeTransform(array, axis)
        return array._with_transform(transform)
    
    @staticmethod
    def flatten(array: 'DynamicArray') -> 'DynamicArray':
        """Flatten array to 1D."""
        transform = FlattenTransform(array)
        return array._with_transform(transform)
    
    @staticmethod
    def pad(array: 'DynamicArray', pad_width: Union[int, Tuple]) -> 'DynamicArray':
        """Pad array."""
        transform = PadTransform(array, pad_width)
        return array._with_transform(transform)
    
    @staticmethod
    def tile(array: 'DynamicArray', reps: Union[int, Tuple]) -> 'DynamicArray':
        """Repeat array along dimensions."""
        transform = TileTransform(array, reps)
        return array._with_transform(transform)
    
    @staticmethod
    def roll(array: 'DynamicArray', shift: int, axis: Optional[int] = None) -> 'DynamicArray':
        """Roll array elements along an axis."""
        transform = RollTransform(array, shift, axis)
        return array._with_transform(transform)
    
    @staticmethod
    def flip(array: 'DynamicArray', axis: int) -> 'DynamicArray':
        """Flip array along an axis."""
        transform = FlipTransform(array, axis)
        return array._with_transform(transform)
    
    @staticmethod
    def clip(array: 'DynamicArray', a_min: Optional[float], a_max: Optional[float]) -> 'DynamicArray':
        """Clip array values to a range."""
        transform = ClipTransform(array, a_min, a_max)
        return array._with_transform(transform)
    
    @staticmethod
    def abs(array: 'DynamicArray') -> 'DynamicArray':
        """Absolute value."""
        transform = AbsTransform(array)
        return array._with_transform(transform)
    
    @staticmethod
    def sign(array: 'DynamicArray') -> 'DynamicArray':
        """Sign of array elements."""
        transform = SignTransform(array)
        return array._with_transform(transform)
    
    @staticmethod
    def round(array: 'DynamicArray', decimals: int = 0) -> 'DynamicArray':
        """Round array elements."""
        transform = RoundTransform(array, decimals)
        return array._with_transform(transform)
    
    @staticmethod
    def sqrt(array: 'DynamicArray') -> 'DynamicArray':
        """Square root."""
        transform = SqrtTransform(array)
        return array._with_transform(transform)
    
    @staticmethod
    def where(condition: 'DynamicArray', x: 'DynamicArray', y: 'DynamicArray') -> 'DynamicArray':
        """Conditional element selection."""
        transform = WhereTransform(condition, x, y)
        return x._with_transform(transform)
    
    @staticmethod
    def multiply(array1: 'DynamicArray', array2: Union['DynamicArray', float]) -> 'DynamicArray':
        """Element-wise multiplication."""
        transform = MultiplyTransform(array1, array2)
        return array1._with_transform(transform)
    
    @staticmethod
    def add(array1: 'DynamicArray', array2: Union['DynamicArray', float]) -> 'DynamicArray':
        """Element-wise addition."""
        transform = AddTransform(array1, array2)
        return array1._with_transform(transform)
    
    @staticmethod
    def min(array: 'DynamicArray', axis: Optional[int] = None) -> 'DynamicArray':
        """Compute minimum along axis (lazy reduction)."""
        transform = MinTransform(array, axis)
        return array._with_transform(transform)
    
    @staticmethod
    def max(array: 'DynamicArray', axis: Optional[int] = None) -> 'DynamicArray':
        """Compute maximum along axis (lazy reduction)."""
        transform = MaxTransform(array, axis)
        return array._with_transform(transform)


def slice_array(array: 'DynamicArray', key) -> 'DynamicArray':
    """Create a lazy slice of an array."""
    transform = SliceTransform(array, key)
    return array._with_transform(transform)
