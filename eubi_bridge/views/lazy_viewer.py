"""
Lazy OME-Zarr Viewer for EuBI-Bridge.

This module provides a memory-efficient viewer for OME-Zarr images that:
- Displays 2D planes with lazy loading (only loads visible chunks)
- Automatically switches to lower resolutions when FOV is too large
- Supports 90-degree rotations to view XY, XZ, or YZ planes
- Provides sliders for navigating through T, C, Z dimensions
- Edge sliders for intuitive panning of the field of view
"""

import streamlit as st
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from eubi_bridge.ngff.multiscales import Pyramid

from eubi_bridge.views.shared import render_path_input

FOV_SIZE_OPTIONS = [128, 256, 512, 1024]
DEFAULT_FOV_SIZE = 256


def render_x_nav(pos_min: int, pos_max: int, current_pos: int, image_width: int = None) -> None:
    """
    Render X-axis horizontal navigation: ◀◀◀ ◀◀ ◀ [X: pos] ▶ ▶▶ ▶▶▶
    Step sizes: 20, 10, 1, 1, 10, 20
    Constrained to image_width if provided.
    """
    btn_width = 40
    pos_width = 70
    total_btn_width = btn_width * 6 + pos_width
    
    if image_width and image_width > total_btn_width:
        padding = (image_width - total_btn_width) // 2
        left_pad, right_pad = padding, padding
    else:
        left_pad, right_pad = 0, 0
    
    col_spec = [left_pad] if left_pad > 0 else []
    col_spec += [btn_width, btn_width, btn_width, pos_width, btn_width, btn_width, btn_width]
    if right_pad > 0:
        col_spec.append(right_pad)
    
    cols = st.columns(col_spec)
    col_offset = 1 if left_pad > 0 else 0
    
    with cols[col_offset + 0]:
        if st.button("◀◀◀", key="x_left_20", help="Move X -20"):
            new_pos = max(pos_min, current_pos - 20)
            st.session_state.viewer_fov_center = (st.session_state.viewer_fov_center[0], new_pos)
            st.rerun()
    
    with cols[col_offset + 1]:
        if st.button("◀◀", key="x_left_10", help="Move X -10"):
            new_pos = max(pos_min, current_pos - 10)
            st.session_state.viewer_fov_center = (st.session_state.viewer_fov_center[0], new_pos)
            st.rerun()
    
    with cols[col_offset + 2]:
        if st.button("◀", key="x_left_1", help="Move X -1"):
            new_pos = max(pos_min, current_pos - 1)
            st.session_state.viewer_fov_center = (st.session_state.viewer_fov_center[0], new_pos)
            st.rerun()
    
    with cols[col_offset + 3]:
        st.markdown(f'<p style="text-align: center; font-weight: bold; font-size: 14px; margin: 8px 0;">X: {current_pos}</p>', 
                    unsafe_allow_html=True)
    
    with cols[col_offset + 4]:
        if st.button("▶", key="x_right_1", help="Move X +1"):
            new_pos = min(pos_max, current_pos + 1)
            st.session_state.viewer_fov_center = (st.session_state.viewer_fov_center[0], new_pos)
            st.rerun()
    
    with cols[col_offset + 5]:
        if st.button("▶▶", key="x_right_10", help="Move X +10"):
            new_pos = min(pos_max, current_pos + 10)
            st.session_state.viewer_fov_center = (st.session_state.viewer_fov_center[0], new_pos)
            st.rerun()
    
    with cols[col_offset + 6]:
        if st.button("▶▶▶", key="x_right_20", help="Move X +20"):
            new_pos = min(pos_max, current_pos + 20)
            st.session_state.viewer_fov_center = (st.session_state.viewer_fov_center[0], new_pos)
            st.rerun()


def render_y_nav_vertical(pos_min: int, pos_max: int, current_pos: int) -> None:
    """
    Render Y-axis vertical navigation stacked top to bottom:
    ▲▲▲ (step 20)
    ▲▲  (step 10)
    ▲   (step 1)
    [Y: pos]
    ▼   (step 1)
    ▼▼  (step 10)
    ▼▼▼ (step 20)
    """
    if st.button("▲▲▲", key="y_up_20", help="Move up 20", use_container_width=True):
        new_pos = max(pos_min, current_pos - 20)
        st.session_state.viewer_fov_center = (new_pos, st.session_state.viewer_fov_center[1])
        st.rerun()
    
    if st.button("▲▲", key="y_up_10", help="Move up 10", use_container_width=True):
        new_pos = max(pos_min, current_pos - 10)
        st.session_state.viewer_fov_center = (new_pos, st.session_state.viewer_fov_center[1])
        st.rerun()
    
    if st.button("▲", key="y_up_1", help="Move up 1", use_container_width=True):
        new_pos = max(pos_min, current_pos - 1)
        st.session_state.viewer_fov_center = (new_pos, st.session_state.viewer_fov_center[1])
        st.rerun()
    
    st.markdown(f'<p style="text-align: center; font-weight: bold; font-size: 12px; margin: 5px 0;">Y: {current_pos}</p>', 
                unsafe_allow_html=True)
    
    if st.button("▼", key="y_down_1", help="Move down 1", use_container_width=True):
        new_pos = min(pos_max, current_pos + 1)
        st.session_state.viewer_fov_center = (new_pos, st.session_state.viewer_fov_center[1])
        st.rerun()
    
    if st.button("▼▼", key="y_down_10", help="Move down 10", use_container_width=True):
        new_pos = min(pos_max, current_pos + 10)
        st.session_state.viewer_fov_center = (new_pos, st.session_state.viewer_fov_center[1])
        st.rerun()
    
    if st.button("▼▼▼", key="y_down_20", help="Move down 20", use_container_width=True):
        new_pos = min(pos_max, current_pos + 20)
        st.session_state.viewer_fov_center = (new_pos, st.session_state.viewer_fov_center[1])
        st.rerun()


MEMORY_BUDGET_MB = 16
ORIENTATIONS = {
    'XY': {'plane_axes': ('y', 'x'), 'slice_axis': 'z', 'display_name': 'XY Plane (top view)'},
    'XZ': {'plane_axes': ('z', 'x'), 'slice_axis': 'y', 'display_name': 'XZ Plane (front view)'},
    'YZ': {'plane_axes': ('y', 'z'), 'slice_axis': 'x', 'display_name': 'YZ Plane (side view)'},
}


@st.cache_resource
def load_pyramid_cached(zarr_path: str) -> Optional[Pyramid]:
    """Load a Pyramid from the given path, cached to avoid repeated I/O."""
    try:
        return Pyramid(zarr_path)
    except Exception as e:
        st.error(f"Failed to load OME-Zarr: {e}")
        return None


def get_axis_index(axes: str, axis_name: str) -> int:
    """Get the index of an axis in the axes string, or -1 if not present."""
    axes_lower = axes.lower()
    return axes_lower.find(axis_name.lower())


def get_dimension_ranges(pyr: Pyramid) -> Dict[str, int]:
    """Get the size of each dimension from the pyramid."""
    axes = pyr.axes.lower()
    shape = pyr.shape
    ranges = {}
    for i, axis in enumerate(axes):
        ranges[axis] = shape[i]
    return ranges


def estimate_memory_mb(height: int, width: int, dtype: np.dtype, channels: int = 1) -> float:
    """Estimate memory usage in MB for a 2D plane."""
    bytes_per_element = dtype.itemsize
    total_bytes = height * width * bytes_per_element * channels
    return total_bytes / (1024 * 1024)


def select_resolution_level(pyr: Pyramid, 
                           plane_height: int, 
                           plane_width: int,
                           memory_budget_mb: float = MEMORY_BUDGET_MB) -> Tuple[str, int, int]:
    """
    Select the appropriate resolution level based on memory budget.
    
    Returns (level_path, adjusted_height, adjusted_width) for the selected level.
    """
    layer0 = pyr.layers['0']
    dtype = layer0.dtype
    
    has_channels = 'c' in pyr.axes.lower()
    num_channels = 1
    if has_channels:
        c_idx = get_axis_index(pyr.axes, 'c')
        num_channels = pyr.shape[c_idx]
    
    resolution_paths = pyr.meta.resolution_paths
    
    for level_path in resolution_paths:
        layer = pyr.layers[level_path]
        layer_shape = layer.shape
        
        scale_factor = pyr.shape[-1] / layer_shape[-1]
        
        scaled_height = int(plane_height / scale_factor)
        scaled_width = int(plane_width / scale_factor)
        
        scaled_height = max(1, min(scaled_height, layer_shape[-2]))
        scaled_width = max(1, min(scaled_width, layer_shape[-1]))
        
        mem_estimate = estimate_memory_mb(scaled_height, scaled_width, dtype, num_channels)
        
        if mem_estimate <= memory_budget_mb:
            return level_path, scaled_height, scaled_width
    
    last_level = resolution_paths[-1]
    layer = pyr.layers[last_level]
    return last_level, layer.shape[-2], layer.shape[-1]


def build_slice_tuple(pyr: Pyramid, 
                      level_path: str,
                      h_axis: str,
                      v_axis: str,
                      indices: Dict[str, int],
                      v_range: Tuple[int, int],
                      h_range: Tuple[int, int]) -> Tuple:
    """
    Build a slice tuple for extracting a 2D plane from the zarr array.
    
    Args:
        pyr: The Pyramid object
        level_path: Resolution level to use
        h_axis: Horizontal axis name (X, Y, or Z)
        v_axis: Vertical axis name (X, Y, or Z)
        indices: Current index for each non-plane dimension (t, c, slice axis, etc.)
        v_range: (start, end) for the vertical axis
        h_range: (start, end) for the horizontal axis
    
    Returns:
        A tuple of slices/indices for array indexing
    """
    layer = pyr.layers[level_path]
    layer_shape = layer.shape
    axes = pyr.axes.lower()
    
    h_axis_lower = h_axis.lower()
    v_axis_lower = v_axis.lower()
    
    scale_factor = pyr.shape[-1] / layer_shape[-1]
    
    slices = []
    
    for i, axis in enumerate(axes):
        if axis == 't':
            slices.append(indices.get('t', 0))
        elif axis == 'c':
            c_idx = indices.get('c', None)
            if c_idx is not None:
                slices.append(c_idx)
            else:
                slices.append(slice(None))
        elif axis == h_axis_lower:
            slices.append(slice(h_range[0], h_range[1]))
        elif axis == v_axis_lower:
            slices.append(slice(v_range[0], v_range[1]))
        elif axis in 'xyz':
            axis_max = layer_shape[i]
            axis_idx = min(int(indices.get(axis, 0) / scale_factor), axis_max - 1)
            slices.append(axis_idx)
    
    return tuple(slices)


def extract_plane(pyr: Pyramid,
                 h_axis: str,
                 v_axis: str,
                 indices: Dict[str, int],
                 fov_center: Tuple[int, int],
                 fov_size: Tuple[int, int],
                 memory_budget_mb: float = MEMORY_BUDGET_MB) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    """
    Extract a 2D plane from the pyramid with automatic resolution selection.
    
    Args:
        pyr: The Pyramid object
        h_axis: Horizontal axis name (X, Y, or Z)
        v_axis: Vertical axis name (X, Y, or Z)
        indices: Current index for each non-plane dimension
        fov_center: (row, col) center of field of view in base resolution
        fov_size: (height, width) of field of view in base resolution
        memory_budget_mb: Maximum memory to use for the plane
    
    Returns:
        (plane_data, level_used, metadata)
    """
    level_path, adj_height, adj_width = select_resolution_level(
        pyr, fov_size[0], fov_size[1], memory_budget_mb
    )
    
    layer = pyr.layers[level_path]
    layer_shape = layer.shape
    base_shape = pyr.shape
    
    scale_factor = base_shape[-1] / layer_shape[-1]
    
    scaled_center = (int(fov_center[0] / scale_factor), int(fov_center[1] / scale_factor))
    scaled_size = (adj_height, adj_width)
    
    half_h = scaled_size[0] // 2
    half_w = scaled_size[1] // 2
    
    v_start = max(0, scaled_center[0] - half_h)
    v_end = min(layer_shape[-2], v_start + scaled_size[0])
    h_start = max(0, scaled_center[1] - half_w)
    h_end = min(layer_shape[-1], h_start + scaled_size[1])
    
    slice_tuple = build_slice_tuple(
        pyr, level_path, h_axis, v_axis, indices,
        (v_start, v_end), (h_start, h_end)
    )
    
    plane_data = layer[slice_tuple]
    
    if isinstance(plane_data, np.ndarray):
        pass
    else:
        plane_data = np.asarray(plane_data)
    
    axes = pyr.axes.lower()
    h_axis_lower = h_axis.lower()
    v_axis_lower = v_axis.lower()
    
    h_pos = axes.find(h_axis_lower)
    v_pos = axes.find(v_axis_lower)
    
    if h_pos >= 0 and v_pos >= 0 and h_pos < v_pos:
        if plane_data.ndim == 2:
            plane_data = plane_data.T
        elif plane_data.ndim == 3:
            plane_data = np.transpose(plane_data, (0, 2, 1))
    
    metadata = {
        'level': level_path,
        'scale_factor': scale_factor,
        'fov_in_level': ((v_start, v_end), (h_start, h_end)),
        'actual_shape': plane_data.shape,
        'h_axis': h_axis,
        'v_axis': v_axis,
    }
    
    return plane_data, level_path, metadata


MAX_DISPLAY_WIDTH = 700
MAX_DISPLAY_HEIGHT = 500


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (255, 255, 255)


def get_channel_display_params(channels: list, channel_idx: int) -> Dict[str, Any]:
    """Extract display parameters for a channel from OME-Zarr metadata."""
    if not channels or channel_idx >= len(channels):
        return {'color': (255, 255, 255), 'window_start': None, 'window_end': None}
    
    ch = channels[channel_idx]
    color_hex = ch.get('color', 'FFFFFF')
    if not color_hex.startswith('#'):
        color_hex = '#' + color_hex
    color_rgb = hex_to_rgb(color_hex)
    
    window = ch.get('window', {})
    window_start = window.get('start')
    window_end = window.get('end')
    
    return {
        'color': color_rgb,
        'window_start': window_start,
        'window_end': window_end,
        'label': ch.get('label', f'Channel {channel_idx}')
    }


def normalize_channel(data: np.ndarray, 
                     window_start: Optional[float] = None, 
                     window_end: Optional[float] = None,
                     percentile_low: float = 1, 
                     percentile_high: float = 99) -> np.ndarray:
    """
    Normalize a single channel using window limits or percentile fallback.
    Returns float array in range [0, 1].
    """
    data = data.astype(np.float32)
    
    if window_start is not None and window_end is not None and window_end > window_start:
        low, high = window_start, window_end
    else:
        low = np.percentile(data, percentile_low)
        high = np.percentile(data, percentile_high)
    
    if high > low:
        data = (data - low) / (high - low)
    else:
        data = np.zeros_like(data)
    
    return np.clip(data, 0, 1)


def normalize_for_display(data: np.ndarray, 
                         channels_meta: Optional[list] = None,
                         channel_indices: Optional[list] = None,
                         user_limits: Optional[Dict[int, Tuple[float, float]]] = None,
                         percentile_low: float = 1, 
                         percentile_high: float = 99) -> np.ndarray:
    """
    Normalize array data to 0-255 uint8 for display using OME-Zarr channel metadata.
    
    Args:
        data: Input image data (2D for single channel, 3D with channels first for multi-channel)
        channels_meta: List of channel metadata dicts from pyr.meta.channels
        channel_indices: Which channels are in the data (for multi-channel composite)
        percentile_low: Fallback percentile for min if no window metadata
        percentile_high: Fallback percentile for max if no window metadata
    
    Returns:
        uint8 array ready for display (grayscale or RGB)
    """
    if data.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    
    if data.ndim == 3 and data.shape[0] <= 4:
        n_channels = data.shape[0]
        height, width = data.shape[1], data.shape[2]
        
        rgb_output = np.zeros((height, width, 3), dtype=np.float32)
        
        for i in range(n_channels):
            if channel_indices is not None and i < len(channel_indices):
                ch_idx = channel_indices[i]
            else:
                ch_idx = i
            
            params = get_channel_display_params(channels_meta, ch_idx)
            
            if user_limits and ch_idx in user_limits:
                window_start, window_end = user_limits[ch_idx]
            else:
                window_start = params['window_start']
                window_end = params['window_end']
            
            normalized = normalize_channel(
                data[i], 
                window_start, 
                window_end,
                percentile_low, 
                percentile_high
            )
            
            color = params['color']
            rgb_output[:, :, 0] += normalized * (color[0] / 255.0)
            rgb_output[:, :, 1] += normalized * (color[1] / 255.0)
            rgb_output[:, :, 2] += normalized * (color[2] / 255.0)
        
        rgb_output = np.clip(rgb_output * 255, 0, 255).astype(np.uint8)
        return rgb_output
    
    if data.ndim > 2:
        data = data.reshape(-1, data.shape[-2], data.shape[-1])[0]
    
    ch_idx = 0
    if channel_indices is not None and len(channel_indices) > 0:
        ch_idx = channel_indices[0]
    
    params = get_channel_display_params(channels_meta, ch_idx)
    
    if user_limits and ch_idx in user_limits:
        window_start, window_end = user_limits[ch_idx]
    else:
        window_start = params['window_start']
        window_end = params['window_end']
    
    normalized = normalize_channel(
        data, 
        window_start, 
        window_end,
        percentile_low, 
        percentile_high
    )
    
    color = params['color']
    if color == (255, 255, 255):
        return (normalized * 255).astype(np.uint8)
    
    height, width = normalized.shape
    rgb_output = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_output[:, :, 0] = (normalized * color[0]).astype(np.uint8)
    rgb_output[:, :, 1] = (normalized * color[1]).astype(np.uint8)
    rgb_output[:, :, 2] = (normalized * color[2]).astype(np.uint8)
    
    return rgb_output


def render_histogram_only(plane_data: np.ndarray, 
                          channels_meta: Optional[list],
                          channel_indices: list):
    """
    Render only the histogram (no sliders).
    Sliders are rendered separately for consistent layout.
    """
    import matplotlib.pyplot as plt
    
    if plane_data.ndim == 2:
        n_channels = 1
        data_list = [plane_data]
    elif plane_data.ndim == 3 and plane_data.shape[0] <= 4:
        n_channels = plane_data.shape[0]
        data_list = [plane_data[i] for i in range(n_channels)]
    else:
        n_channels = 1
        data_list = [plane_data.reshape(-1, plane_data.shape[-2], plane_data.shape[-1])[0]]
    
    fig, ax = plt.subplots(figsize=(3, 2.5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    legend_labels = []
    for i, ch_data in enumerate(data_list):
        if i < len(channel_indices):
            ch_idx = channel_indices[i]
        else:
            ch_idx = i
        
        params = get_channel_display_params(channels_meta, ch_idx)
        color_rgb = params['color']
        color_hex = '#{:02x}{:02x}{:02x}'.format(*color_rgb)
        label = params.get('label', f'Ch {ch_idx}')
        
        if color_rgb == (255, 255, 255):
            hist_color = '#888888'
        else:
            hist_color = color_hex
        
        flat_data = ch_data.flatten()
        ax.hist(flat_data, bins=64, color=hist_color, alpha=0.6, 
                edgecolor='none', label=f"{label}")
        legend_labels.append((ch_idx, label, hist_color))
    
    ax.tick_params(axis='both', labelsize=7, colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if len(legend_labels) > 1:
        ax.legend(fontsize=6, loc='upper right', 
                 facecolor='#0e1117', edgecolor='white', labelcolor='white')
    
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_intensity_sliders(plane_data: np.ndarray, 
                             channels_meta: Optional[list],
                             channel_indices: list) -> Dict[int, Tuple[float, float]]:
    """
    Render intensity sliders for each channel.
    Call this separately for consistent slider layout.
    
    Returns:
        Dictionary mapping channel index to (min, max) intensity limits from slider values
    """
    user_limits = {}
    
    if plane_data.ndim == 2:
        data_list = [plane_data]
    elif plane_data.ndim == 3 and plane_data.shape[0] <= 4:
        data_list = [plane_data[i] for i in range(plane_data.shape[0])]
    else:
        data_list = [plane_data.reshape(-1, plane_data.shape[-2], plane_data.shape[-1])[0]]
    
    if 'viewer_intensity_limits' not in st.session_state:
        st.session_state.viewer_intensity_limits = {}
    
    for i, ch_data in enumerate(data_list):
        if i < len(channel_indices):
            ch_idx = channel_indices[i]
        else:
            ch_idx = i
        
        params = get_channel_display_params(channels_meta, ch_idx)
        label = params.get('label', f'Channel {ch_idx}')
        color_rgb = params['color']
        color_hex = '#{:02x}{:02x}{:02x}'.format(*color_rgb)
        
        window_start = params.get('window_start')
        window_end = params.get('window_end')
        
        ch_min = float(np.min(ch_data))
        ch_max = float(np.max(ch_data))
        
        state_key = f"intensity_{ch_idx}"
        if state_key not in st.session_state.viewer_intensity_limits:
            if window_start is not None and window_end is not None:
                default_min = max(ch_min, float(window_start))
                default_max = min(ch_max, float(window_end))
            else:
                default_min = ch_min
                default_max = ch_max
            st.session_state.viewer_intensity_limits[state_key] = (default_min, default_max)
        
        current_limits = st.session_state.viewer_intensity_limits[state_key]
        
        if ch_max > ch_min:
            new_limits = st.slider(
                f"Ch {ch_idx}: {label}",
                min_value=ch_min,
                max_value=ch_max,
                value=(max(ch_min, current_limits[0]), min(ch_max, current_limits[1])),
                key=f"intensity_slider_{ch_idx}"
            )
            st.session_state.viewer_intensity_limits[state_key] = new_limits
            user_limits[ch_idx] = new_limits
        else:
            user_limits[ch_idx] = (ch_min, ch_max)
    
    return user_limits


def render_combined_histogram(plane_data: np.ndarray, 
                              channels_meta: Optional[list],
                              channel_indices: list) -> Dict[int, Tuple[float, float]]:
    """
    Render a combined histogram with all channels overlaid.
    Includes sliders for each channel to adjust intensity limits.
    
    Returns:
        Dictionary mapping channel index to (min, max) intensity limits set by user
    """
    render_histogram_only(plane_data, channels_meta, channel_indices)
    return {}


def fit_to_display(data: np.ndarray, 
                   max_width: int = MAX_DISPLAY_WIDTH, 
                   max_height: int = MAX_DISPLAY_HEIGHT) -> Tuple[np.ndarray, float]:
    """
    Scale image to fit within max dimensions while preserving aspect ratio.
    
    Returns:
        (scaled_data, scale_factor) where scale_factor is how much the image was scaled
    """
    if data.ndim == 2:
        h, w = data.shape
    elif data.ndim == 3:
        h, w = data.shape[:2]
    else:
        return data, 1.0
    
    if h == 0 or w == 0:
        return data, 1.0
    
    scale_w = max_width / w if w > max_width else 1.0
    scale_h = max_height / h if h > max_height else 1.0
    scale = min(scale_w, scale_h)
    
    if scale >= 1.0:
        return data, 1.0
    
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    
    from PIL import Image
    
    if data.ndim == 2:
        img = Image.fromarray(data)
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return np.array(img_resized), scale
    else:
        img = Image.fromarray(data)
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return np.array(img_resized), scale


def get_plane_dimensions(pyr: Pyramid, h_axis: str, v_axis: str) -> Tuple[int, int]:
    """Get the dimensions of the viewing plane at base resolution."""
    axes = pyr.axes.lower()
    shape = pyr.shape
    dim_sizes = dict(zip(axes, shape))
    
    width = dim_sizes.get(h_axis.lower(), shape[-1])
    height = dim_sizes.get(v_axis.lower(), shape[-2])
    
    return height, width


def init_viewer_state():
    """Initialize viewer state in session state."""
    if 'viewer_h_axis' not in st.session_state:
        st.session_state.viewer_h_axis = 'X'
    if 'viewer_v_axis' not in st.session_state:
        st.session_state.viewer_v_axis = 'Y'
    if 'viewer_indices' not in st.session_state:
        st.session_state.viewer_indices = {}
    if 'viewer_fov_center' not in st.session_state:
        st.session_state.viewer_fov_center = None
    if 'viewer_fov_size' not in st.session_state:
        st.session_state.viewer_fov_size = None
    if 'viewer_zoom' not in st.session_state:
        st.session_state.viewer_zoom = 1.0


def reset_viewer_state_for_new_dataset():
    """Reset viewer state when loading a new dataset."""
    st.session_state.viewer_h_axis = 'X'
    st.session_state.viewer_v_axis = 'Y'
    st.session_state.viewer_indices = {}
    st.session_state.viewer_fov_center = None
    st.session_state.viewer_fov_size = None
    st.session_state.viewer_zoom = 1.0
    if 'viewer_selected_channels' in st.session_state:
        del st.session_state.viewer_selected_channels
    if 'viewer_intensity_limits' in st.session_state:
        del st.session_state.viewer_intensity_limits


def render(bridge=None):
    """Render the lazy OME-Zarr viewer interface."""
    st.subheader("OME-Zarr Viewer")
    st.markdown("View 2D planes of OME-Zarr images with lazy loading and automatic resolution switching.")
    
    init_viewer_state()
    
    zarr_path = render_path_input(
        key_prefix="viewer",
        label="OME-Zarr Path",
        help_text="Select an OME-Zarr file or folder to view"
    )
    
    if not zarr_path:
        st.info("Please select an OME-Zarr file to view.")
        return
    
    zarr_path_obj = Path(zarr_path)
    if not zarr_path_obj.exists():
        st.warning("Selected path does not exist.")
        return
    
    if 'viewer_current_dataset' not in st.session_state:
        st.session_state.viewer_current_dataset = None
    
    if st.session_state.viewer_current_dataset != zarr_path:
        reset_viewer_state_for_new_dataset()
        st.session_state.viewer_current_dataset = zarr_path
        st.rerun()
    
    pyr = load_pyramid_cached(zarr_path)
    if pyr is None:
        return
    
    st.markdown("---")
    
    with st.expander("OME-Zarr Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Name:** {zarr_path_obj.name}")
            st.markdown(f"**Axes:** {pyr.axes}")
            st.markdown(f"**Shape:** {pyr.shape}")
        with col2:
            st.markdown(f"**Data Type:** {pyr.layers['0'].dtype}")
            st.markdown(f"**Resolution Levels:** {pyr.nlayers}")
    
    st.markdown("---")
    st.markdown("**Viewing Controls**")
    
    axes = pyr.axes.lower()
    dim_ranges = get_dimension_ranges(pyr)
    
    spatial_axes = [a.upper() for a in axes if a in 'xyz']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        prev_h_axis = st.session_state.viewer_h_axis
        h_axis = st.selectbox(
            "Horizontal Axis",
            options=spatial_axes,
            index=spatial_axes.index(st.session_state.viewer_h_axis) 
                  if st.session_state.viewer_h_axis in spatial_axes else 0,
            key="h_axis_select"
        )
        if h_axis != prev_h_axis:
            st.session_state.viewer_fov_center = None
            st.session_state.viewer_fov_size = None
        st.session_state.viewer_h_axis = h_axis
    
    with col2:
        available_v_axes = [a for a in spatial_axes if a != h_axis]
        if st.session_state.viewer_v_axis not in available_v_axes:
            st.session_state.viewer_v_axis = available_v_axes[0] if available_v_axes else 'Y'
        prev_v_axis = st.session_state.viewer_v_axis
        v_axis = st.selectbox(
            "Vertical Axis",
            options=available_v_axes,
            index=available_v_axes.index(st.session_state.viewer_v_axis) 
                  if st.session_state.viewer_v_axis in available_v_axes else 0,
            key="v_axis_select"
        )
        if v_axis != prev_v_axis:
            st.session_state.viewer_fov_center = None
            st.session_state.viewer_fov_size = None
        st.session_state.viewer_v_axis = v_axis
    
    with col3:
        if 't' in axes:
            t_max = dim_ranges['t'] - 1
            t_idx = st.slider("Time (T)", 0, t_max, 
                            st.session_state.viewer_indices.get('t', 0),
                            key="t_slider")
            st.session_state.viewer_indices['t'] = t_idx
    
    with col4:
        if 'c' in axes:
            num_channels = dim_ranges['c']
            channels_meta = getattr(pyr.meta, 'channels', None)
            
            if 'viewer_selected_channels' not in st.session_state:
                st.session_state.viewer_selected_channels = list(range(num_channels))
            
            with st.expander("Channels", expanded=False):
                col_all, col_none = st.columns(2)
                with col_all:
                    if st.button("All", key="ch_select_all", use_container_width=True):
                        st.session_state.viewer_selected_channels = list(range(num_channels))
                        st.rerun()
                with col_none:
                    if st.button("None", key="ch_select_none", use_container_width=True):
                        st.session_state.viewer_selected_channels = []
                        st.rerun()
                
                for ch_idx in range(num_channels):
                    if channels_meta and ch_idx < len(channels_meta):
                        ch_info = channels_meta[ch_idx]
                        label = ch_info.get('label', f'Ch {ch_idx}')
                        color = ch_info.get('color', 'FFFFFF')
                        if not color.startswith('#'):
                            color = '#' + color
                    else:
                        label = f'Ch {ch_idx}'
                        color = '#FFFFFF'
                    
                    is_selected = ch_idx in st.session_state.viewer_selected_channels
                    
                    col_cb, col_swatch = st.columns([4, 1])
                    with col_cb:
                        checked = st.checkbox(
                            f"{label}",
                            value=is_selected,
                            key=f"ch_checkbox_{ch_idx}"
                        )
                    with col_swatch:
                        st.markdown(
                            f'<div style="background-color:{color};width:20px;height:20px;'
                            f'border-radius:3px;border:1px solid #ccc;margin-top:5px;"></div>',
                            unsafe_allow_html=True
                        )
                    
                    if checked and ch_idx not in st.session_state.viewer_selected_channels:
                        st.session_state.viewer_selected_channels.append(ch_idx)
                        st.session_state.viewer_selected_channels.sort()
                    elif not checked and ch_idx in st.session_state.viewer_selected_channels:
                        st.session_state.viewer_selected_channels.remove(ch_idx)
            
            selected = st.session_state.viewer_selected_channels
            if len(selected) == 0:
                st.session_state.viewer_indices['c'] = 0
                st.session_state.viewer_selected_channels = [0]
            elif len(selected) == 1:
                st.session_state.viewer_indices['c'] = selected[0]
            else:
                st.session_state.viewer_indices['c'] = None
    
    slice_axis = [a for a in spatial_axes if a != h_axis and a != v_axis]
    slice_axis = slice_axis[0].lower() if slice_axis else None
    
    st.markdown("---")
    
    plane_height, plane_width = get_plane_dimensions(pyr, h_axis, v_axis)
    
    if st.session_state.viewer_fov_center is None:
        st.session_state.viewer_fov_center = (plane_height // 2, plane_width // 2)
        st.session_state.viewer_fov_size = (DEFAULT_FOV_SIZE, DEFAULT_FOV_SIZE)
    
    fov_height, fov_width = st.session_state.viewer_fov_size
    fov_height = min(fov_height, plane_height)
    fov_width = min(fov_width, plane_width)
    st.session_state.viewer_fov_size = (fov_height, fov_width)
    
    center_row, center_col = st.session_state.viewer_fov_center
    center_row = max(0, min(center_row, plane_height - 1))
    center_col = max(0, min(center_col, plane_width - 1))
    st.session_state.viewer_fov_center = (center_row, center_col)
    
    show_v_nav = plane_height > fov_height
    show_h_nav = plane_width > fov_width
    
    half_fov_h = fov_height // 2
    half_fov_w = fov_width // 2
    v_min = half_fov_h
    v_max = plane_height - half_fov_h - 1
    h_min = half_fov_w
    h_max = plane_width - half_fov_w - 1
    
    has_slice_slider = slice_axis is not None and slice_axis in dim_ranges
    has_position_sliders = (show_h_nav and h_max > h_min) or (show_v_nav and v_max > v_min)
    
    if has_slice_slider:
        slice_max = dim_ranges[slice_axis] - 1
        current_slice = st.session_state.viewer_indices.get(slice_axis, slice_max // 2)
        slice_idx = st.slider(
            f"{slice_axis.upper()} Slice",
            0, slice_max,
            current_slice,
            key="slice_slider"
        )
        st.session_state.viewer_indices[slice_axis] = slice_idx
    
    if has_position_sliders:
        if show_h_nav and h_max > h_min:
            new_h = st.slider(
                f"{h_axis} Position",
                min_value=h_min,
                max_value=h_max,
                value=center_col,
                key="h_pos_slider"
            )
            if new_h != center_col:
                center_col = new_h
                st.session_state.viewer_fov_center = (center_row, center_col)
        
        if show_v_nav and v_max > v_min:
            new_v = st.slider(
                f"{v_axis} Position",
                min_value=v_min,
                max_value=v_max,
                value=center_row,
                key="v_pos_slider"
            )
            if new_v != center_row:
                center_row = new_v
                st.session_state.viewer_fov_center = (center_row, center_col)
    
    st.markdown("**Image View**")
    
    try:
        plane_data, level_used, meta = extract_plane(
            pyr,
            h_axis,
            v_axis,
            st.session_state.viewer_indices,
            st.session_state.viewer_fov_center,
            st.session_state.viewer_fov_size,
            MEMORY_BUDGET_MB
        )
        
        channels_meta = getattr(pyr.meta, 'channels', None)
        
        if 'c' in dim_ranges:
            channel_indices = st.session_state.get('viewer_selected_channels', list(range(dim_ranges['c'])))
            if not channel_indices:
                channel_indices = [0]
            
            if plane_data.ndim == 3 and plane_data.shape[0] > 1:
                num_extracted_channels = plane_data.shape[0]
                valid_indices = [idx for idx in channel_indices if idx < num_extracted_channels]
                if valid_indices:
                    plane_data = plane_data[valid_indices]
                    channel_indices = valid_indices
        else:
            channel_indices = [0]
        
        if 'viewer_intensity_limits' not in st.session_state:
            st.session_state.viewer_intensity_limits = {}
        
        img_col, hist_col = st.columns([3, 1])
        
        with hist_col:
            render_combined_histogram(plane_data, channels_meta, channel_indices)
            user_limits = render_intensity_sliders(plane_data, channels_meta, channel_indices)
        
        display_data = normalize_for_display(plane_data, channels_meta, channel_indices, user_limits)
        
        actual_width = display_data.shape[1] if len(display_data.shape) >= 2 else fov_width
        actual_height = display_data.shape[0] if len(display_data.shape) >= 2 else fov_height
        shape_dict = f"({h_axis.lower()}:{actual_width}, {v_axis.lower()}:{actual_height})"
        
        fitted_data, display_scale = fit_to_display(display_data)
        fitted_w = fitted_data.shape[1] if len(fitted_data.shape) >= 2 else actual_width
        fitted_h = fitted_data.shape[0] if len(fitted_data.shape) >= 2 else actual_height
        
        if display_scale < 1.0:
            info_text = f"Level: {level_used} | Scale: {meta['scale_factor']:.1f}x | Shape: {shape_dict} | Display: {display_scale:.0%}"
        else:
            info_text = f"Level: {level_used} | Scale: {meta['scale_factor']:.1f}x | Shape: {shape_dict}"
        
        with img_col:
            st.caption(info_text)
            st.image(fitted_data)
        
    except Exception as e:
        st.error(f"Error extracting plane: {e}")
        import traceback
        st.code(traceback.format_exc())
        return
    
    st.markdown("---")
    st.markdown("**Field of View Settings**")
    
    fov_col1, fov_col2 = st.columns(2)
    
    available_v_sizes = [s for s in FOV_SIZE_OPTIONS if s <= plane_height]
    if plane_height not in available_v_sizes:
        available_v_sizes.append(plane_height)
    available_v_sizes = sorted(available_v_sizes)
    
    available_h_sizes = [s for s in FOV_SIZE_OPTIONS if s <= plane_width]
    if plane_width not in available_h_sizes:
        available_h_sizes.append(plane_width)
    available_h_sizes = sorted(available_h_sizes)
    
    with fov_col1:
        if len(available_h_sizes) == 1:
            st.selectbox(
                f"FOV {h_axis} Size (full)",
                options=[available_h_sizes[0]],
                index=0,
                disabled=True,
                key="fov_h_size_disabled"
            )
            new_fov_width = available_h_sizes[0]
        else:
            current_h_idx = 0
            if fov_width in available_h_sizes:
                current_h_idx = available_h_sizes.index(fov_width)
            else:
                closest = min(available_h_sizes, key=lambda x: abs(x - fov_width))
                current_h_idx = available_h_sizes.index(closest)
            
            new_fov_width = st.selectbox(
                f"FOV {h_axis} Size",
                options=available_h_sizes,
                index=current_h_idx,
                key="fov_h_size"
            )
    
    with fov_col2:
        if len(available_v_sizes) == 1:
            st.selectbox(
                f"FOV {v_axis} Size (full)",
                options=[available_v_sizes[0]],
                index=0,
                disabled=True,
                key="fov_v_size_disabled"
            )
            new_fov_height = available_v_sizes[0]
        else:
            current_v_idx = 0
            if fov_height in available_v_sizes:
                current_v_idx = available_v_sizes.index(fov_height)
            else:
                closest = min(available_v_sizes, key=lambda x: abs(x - fov_height))
                current_v_idx = available_v_sizes.index(closest)
            
            new_fov_height = st.selectbox(
                f"FOV {v_axis} Size",
                options=available_v_sizes,
                index=current_v_idx,
                key="fov_v_size"
            )
    
    if (new_fov_height, new_fov_width) != st.session_state.viewer_fov_size:
        st.session_state.viewer_fov_size = (new_fov_height, new_fov_width)
        st.rerun()
    
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("Reset View", key="reset_view"):
            st.session_state.viewer_fov_center = (plane_height // 2, plane_width // 2)
            st.session_state.viewer_fov_size = (min(DEFAULT_FOV_SIZE, plane_height), min(DEFAULT_FOV_SIZE, plane_width))
            st.rerun()
    
    with btn_col2:
        if st.button("View Full Image", key="full_view"):
            st.session_state.viewer_fov_center = (plane_height // 2, plane_width // 2)
            st.session_state.viewer_fov_size = (plane_height, plane_width)
            st.rerun()
    
    with st.expander("View Metadata"):
        st.json({
            'horizontal_axis': h_axis,
            'vertical_axis': v_axis,
            'slice_axis': slice_axis,
            'indices': st.session_state.viewer_indices,
            'fov_center': st.session_state.viewer_fov_center,
            'fov_size': st.session_state.viewer_fov_size,
            'level_used': level_used,
            'scale_factor': meta['scale_factor'],
            'actual_shape': list(meta['actual_shape']),
        })
