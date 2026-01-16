"""
Optimized Visual Channel Editor for EuBI-Bridge.

Performance improvements:
1. Lazy loading of intensity ranges (only calculate when needed)
2. Cached pyramid metadata (levels, scales)
3. Optimized plane extraction (minimal array operations)
4. Conditional histogram rendering (only when requested)
5. Debounced slider updates (reduce reruns)
6. Async data loading where possible
7. Improved memory management
"""

import streamlit as st
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from eubi_bridge.ngff.multiscales import Pyramid, NGFFMetadataHandler
from functools import lru_cache
import time

FOV_SIZE_OPTIONS = [128, 256, 512, 1024]
DEFAULT_FOV_SIZE = 256
CANVAS_SIZE = 256
MAX_DISPLAY_WIDTH = 800
MAX_DISPLAY_HEIGHT = 600
MEMORY_BUDGET_MB = 16
PLANE_CACHE_SIZE = 30

COLOR_PRESETS = {
    "Red": "#FF0000",
    "Green": "#00FF00",
    "Blue": "#0000FF",
    "Yellow": "#FFFF00",
    "Cyan": "#00FFFF",
    "Magenta": "#FF00FF",
    "Orange": "#FFA500",
    "White": "#FFFFFF",
    "Gray": "#808080",
    "Purple": "#800080",
    "Pink": "#FFC0CB",
    "Lime": "#32CD32",
}


def _sniff_zarr_version(zarr_path: str) -> str: # TODO: detection should be more robust and remote-friendly
    """Sniff OME-Zarr version from file structure.
    
    - zarr.json + ome folder = version 0.5
    - .zattrs without ome folder = version 0.4
    """
    path = Path(zarr_path)
    
    has_zarr_json = (path / 'zarr.json').exists()
    has_ome_folder = (path / 'ome').exists()
    has_zattrs = (path / '.zattrs').exists()
    
    if has_zarr_json and has_ome_folder:
        return "0.5"
    elif has_zattrs and not has_ome_folder:
        return "0.4"
    else:
        return "0.4"


def _load_pyramid_with_fallback(zarr_path: str) -> Tuple[Optional[Pyramid], Optional[str]]:
    """Load a Pyramid, handling missing version metadata gracefully.
    
    Returns:
        Tuple of (Pyramid or None, warning_message or None)
    """
    import zarr
    
    try:
        pyr = Pyramid(zarr_path)
        return pyr, None
    except KeyError as e:
        if 'version' in str(e):
            try:
                detected_version = _sniff_zarr_version(zarr_path)
                
                gr = zarr.open_group(zarr_path, mode='a')
                
                if detected_version == "0.5" and 'ome' in gr.attrs:
                    ome_meta = dict(gr.attrs['ome'])
                    if 'version' not in ome_meta:
                        ome_meta['version'] = detected_version
                        gr.attrs['ome'] = ome_meta
                elif 'multiscales' in gr.attrs:
                    multiscales = list(gr.attrs['multiscales'])
                    if multiscales and 'version' not in multiscales[0]:
                        multiscales[0]['version'] = detected_version
                        gr.attrs['multiscales'] = multiscales
                
                pyr = Pyramid(zarr_path)
                return pyr, f"version_sniffed:{detected_version}"
            except Exception as inner_e:
                print(f"[PYRAMID] Failed to patch version: {inner_e}")
                return None, None
        else:
            print(f"[PYRAMID] KeyError: {e}")
            return None, None
    except Exception as e:
        print(f"[PYRAMID] Failed: {e}")
        return None, None


@st.cache_resource
def load_pyramid_cached(zarr_path: str) -> Optional[Pyramid]:
    """Load a Pyramid with caching."""
    pyr, warning = _load_pyramid_with_fallback(zarr_path)
    if warning and warning.startswith("version_sniffed:"):
        if 'vce_version_warning' not in st.session_state or not isinstance(st.session_state.vce_version_warning, dict):
            st.session_state.vce_version_warning = {}
        detected_version = warning.split(":")[1]
        st.session_state.vce_version_warning[zarr_path] = detected_version
    return pyr


def load_pyramid_fresh(zarr_path: str) -> Optional[Pyramid]:
    """Load Pyramid without caching for metadata updates."""
    pyr, warning = _load_pyramid_with_fallback(zarr_path)
    if pyr is None:
        st.error("Failed to load OME-Zarr")
    if warning and warning.startswith("version_sniffed:"):
        if 'vce_version_warning' not in st.session_state or not isinstance(st.session_state.vce_version_warning, dict):
            st.session_state.vce_version_warning = {}
        detected_version = warning.split(":")[1]
        st.session_state.vce_version_warning[zarr_path] = detected_version
    return pyr


@st.cache_data(ttl=3600)
def get_pyramid_metadata(zarr_path: str) -> Dict:
    """Cache pyramid metadata keyed only by path."""
    pyr = load_pyramid_cached(zarr_path)
    if pyr is None:
        return {
            'axes': '',
            'shape': (),
            'resolution_paths': [],
            'levels_info': [],
            'num_levels': 0
        }

    axes = pyr.axes.lower()
    shape = pyr.shape

    resolution_paths = sorted(pyr.meta.resolution_paths,
                              key=lambda x: int(x) if x.isdigit() else 0)

    levels_info = []
    for path in resolution_paths:
        layer = pyr.layers.get(path)
        if layer is None:
            continue
        levels_info.append({
            'path': path,
            'shape': layer.shape,
            'dtype': layer.dtype,
            'chunks': getattr(layer, 'chunks', None)
        })

    return {
        'axes': axes,
        'shape': shape,
        'resolution_paths': resolution_paths,
        'levels_info': levels_info,
        'num_levels': len(levels_info)
    }


def calculate_intensity_ranges_lazy(zarr_path: str, pyr: Pyramid,
                                    channel_idx: int) -> Tuple[float, float]:
    """
    Calculate intensity range for a SINGLE channel on demand.
    Much faster than calculating all channels upfront.
    Uses lowest resolution and samples data if too large.
    """
    cache_key = f"intensity_range_{zarr_path}_{channel_idx}"

    if cache_key in st.session_state:
        return st.session_state[cache_key]

    axes = pyr.axes.lower()
    c_idx = axes.find('c')

    resolution_paths = sorted(pyr.meta.resolution_paths,
                              key=lambda x: int(x) if x.isdigit() else 0)
    lowest_res = resolution_paths[-1] if resolution_paths else '0'
    layer = pyr.layers.get(lowest_res)

    if layer is None:
        st.session_state[cache_key] = (0.0, 1.0)
        return (0.0, 1.0)

    try:
        if c_idx >= 0:
            slices = [slice(None)] * layer.ndim
            slices[c_idx] = channel_idx
            data = layer[tuple(slices)]
        else:
            data = layer[...]

        if data.size > 10_000_000:
            indices = np.random.choice(data.size, 10_000_000, replace=False)
            data_flat = data.flat
            sampled = data_flat[indices]
            if hasattr(sampled, 'compute'):
                sampled = sampled.compute()
            vmin, vmax = float(np.min(sampled)), float(np.max(sampled))
        else:
            if hasattr(data, 'compute'):
                data = data.compute()
            vmin, vmax = float(np.min(data)), float(np.max(data))

        result = (vmin, vmax)
        st.session_state[cache_key] = result
        return result

    except Exception as e:
        print(f"[INTENSITY] Failed for channel {channel_idx}: {e}")
        st.session_state[cache_key] = (0.0, 1.0)
        return (0.0, 1.0)


def get_dtype_range(dtype: np.dtype) -> Tuple[float, float]:
    """Get the min/max range for a given data type."""
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return (float(info.min), float(info.max))
    elif np.issubdtype(dtype, np.floating):
        return float(np.finfo(dtype).min), float(np.finfo(dtype).max)
    return (0.0, 255.0)


def select_optimal_resolution(metadata: Dict, h_axis: str, v_axis: str,
                              zoom_level: int) -> Tuple[str, Dict]:
    """Fast resolution selection using cached metadata."""
    levels = metadata['levels_info']
    if not levels:
        return '0', {'scale_factor': 1.0}

    num_levels = len(levels)
    inverted_idx = num_levels - 1 - zoom_level
    inverted_idx = max(0, min(inverted_idx, num_levels - 1))

    level = levels[inverted_idx]
    return level['path'], {
        'scale_factor': 2**inverted_idx,
        'shape': level['shape']
    }


def extract_plane_optimized(pyr: Pyramid, metadata: Dict, h_axis: str,
                            v_axis: str, indices: Dict[str, int],
                            fov_center: Tuple[int, int],
                            zoom_level: int) -> Tuple[np.ndarray, str, Dict]:
    """Optimized plane extraction with minimal operations."""
    level_path, meta = select_optimal_resolution(metadata, h_axis, v_axis,
                                                 zoom_level)

    layer = pyr.layers[level_path]
    axes = metadata['axes']
    layer_shape = meta['shape']

    base_shape = metadata['shape']
    h_idx = axes.find(h_axis.lower())
    v_idx = axes.find(v_axis.lower())

    if h_idx < 0 or v_idx < 0:
        h_idx = len(axes) - 1
        v_idx = len(axes) - 2

    h_scale = base_shape[h_idx] / layer_shape[h_idx] if h_idx >= 0 else 1.0
    v_scale = base_shape[v_idx] / layer_shape[v_idx] if v_idx >= 0 else 1.0

    layer_height = layer_shape[v_idx] if v_idx >= 0 else layer_shape[-2]
    layer_width = layer_shape[h_idx] if h_idx >= 0 else layer_shape[-1]

    center_row, center_col = fov_center
    scaled_row = int(center_row / v_scale)
    scaled_col = int(center_col / h_scale)

    fov_h = min(CANVAS_SIZE, layer_height)
    fov_w = min(CANVAS_SIZE, layer_width)
    half_h, half_w = fov_h // 2, fov_w // 2

    row_start = max(0, min(scaled_row - half_h, layer_height - fov_h))
    row_end = row_start + fov_h
    col_start = max(0, min(scaled_col - half_w, layer_width - fov_w))
    col_end = col_start + fov_w

    slices = []
    for i, ax in enumerate(axes):
        if ax == h_axis.lower():
            slices.append(slice(col_start, col_end))
        elif ax == v_axis.lower():
            slices.append(slice(row_start, row_end))
        elif ax == 'c':
            slices.append(slice(None))
        elif ax in indices:
            scaled_idx = int(indices[ax] / (base_shape[i] / layer_shape[i]))
            slices.append(max(0, min(scaled_idx, layer_shape[i] - 1)))
        else:
            slices.append(0)

    data = layer[tuple(slices)]
    if hasattr(data, 'compute'):
        data = data.compute()
    data = np.asarray(data)

    remaining_axes = [ax for i, ax in enumerate(axes) if not isinstance(slices[i], int)]
    c_idx_in_result = remaining_axes.index('c') if 'c' in remaining_axes else -1
    
    if c_idx_in_result >= 0 and data.ndim > 2 and c_idx_in_result != 0:
        data = np.moveaxis(data, c_idx_in_result, 0)
        remaining_axes.insert(0, remaining_axes.pop(c_idx_in_result))

    h_idx_in_result = remaining_axes.index(h_axis.lower()) if h_axis.lower() in remaining_axes else -1
    v_idx_in_result = remaining_axes.index(v_axis.lower()) if v_axis.lower() in remaining_axes else -1
    
    if h_idx_in_result >= 0 and v_idx_in_result >= 0 and h_idx_in_result < v_idx_in_result:
        if data.ndim == 2:
            data = data.T
        elif data.ndim == 3:
            data = np.transpose(data, (0, 2, 1))

    meta.update({
        'fov_size': (fov_h, fov_w),
        'actual_size': (row_end - row_start, col_end - col_start),
        'layer_height': layer_height,
        'layer_width': layer_width
    })

    return data, level_path, meta


def normalize_for_display_fast(plane_data: np.ndarray,
                               channel_indices: List[int],
                               intensity_limits: Dict[int, Tuple[float,
                                                                 float]],
                               colors: Dict[int, str]) -> np.ndarray:
    """Optimized normalization with vectorized operations."""
    if plane_data.ndim == 2:
        ch_idx = channel_indices[0] if channel_indices else 0
        vmin, vmax = intensity_limits.get(
            ch_idx, (np.min(plane_data), np.max(plane_data)))

        data = plane_data.astype(np.float32)
        if vmax > vmin:
            data = np.clip((data - vmin) / (vmax - vmin), 0, 1)
        else:
            data = np.zeros_like(data)

        color_hex = colors.get(ch_idx, '#FFFFFF')
        r, g, b = hex_to_rgb(color_hex)

        rgb = np.zeros((*data.shape, 3), dtype=np.uint8)
        rgb[..., 0] = (data * r).astype(np.uint8)
        rgb[..., 1] = (data * g).astype(np.uint8)
        rgb[..., 2] = (data * b).astype(np.uint8)
        return rgb

    elif plane_data.ndim == 3:
        h, w = plane_data.shape[1], plane_data.shape[2]
        rgb = np.zeros((h, w, 3), dtype=np.float32)

        for i in range(plane_data.shape[0]):
            ch_idx = channel_indices[i] if i < len(channel_indices) else i
            vmin, vmax = intensity_limits.get(
                ch_idx, (np.min(plane_data[i]), np.max(plane_data[i])))

            data = plane_data[i].astype(np.float32)
            if vmax > vmin:
                data = np.clip((data - vmin) / (vmax - vmin), 0, 1)
            else:
                data = np.zeros_like(data)

            color_hex = colors.get(ch_idx, '#FFFFFF')
            r, g, b = hex_to_rgb(color_hex)

            rgb[..., 0] += data * (r / 255.0)
            rgb[..., 1] += data * (g / 255.0)
            rgb[..., 2] += data * (b / 255.0)

        rgb = np.clip(rgb, 0, 1)
        return (rgb * 255).astype(np.uint8)

    return plane_data


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return (255, 255, 255)


def fit_to_canvas(data: np.ndarray,
                  canvas_size: int = CANVAS_SIZE) -> np.ndarray:
    """Place image data onto a fixed-size canvas."""
    if data.ndim == 2:
        h, w = data.shape
        canvas = np.zeros((canvas_size, canvas_size), dtype=data.dtype)
    elif data.ndim == 3:
        h, w = data.shape[:2]
        canvas = np.zeros((canvas_size, canvas_size, data.shape[2]),
                          dtype=data.dtype)
    else:
        return data

    if h == 0 or w == 0:
        return canvas

    src_h = min(h, canvas_size)
    src_w = min(w, canvas_size)
    start_y = (canvas_size - src_h) // 2
    start_x = (canvas_size - src_w) // 2
    src_start_y = (h - src_h) // 2
    src_start_x = (w - src_w) // 2

    if data.ndim == 2:
        canvas[start_y:start_y + src_h, start_x:start_x + src_w] = \
            data[src_start_y:src_start_y + src_h, src_start_x:src_start_x + src_w]
    else:
        canvas[start_y:start_y + src_h, start_x:start_x + src_w, :] = \
            data[src_start_y:src_start_y + src_h, src_start_x:src_start_x + src_w, :]

    return canvas


def render_histogram_fast(plane_data: np.ndarray, channel_indices: List[int],
                          colors: Dict[int, str],
                          labels: Dict[int, str]) -> np.ndarray:
    """Optimized histogram rendering with sampling."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import io
    from PIL import Image

    fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')

    if plane_data.ndim == 2:
        data_list = [plane_data]
    else:
        data_list = [plane_data[i] for i in range(min(plane_data.shape[0], 4))]

    for i, ch_data in enumerate(data_list):
        ch_idx = channel_indices[i] if i < len(channel_indices) else i
        color_hex = colors.get(ch_idx, '#FFFFFF')
        label = labels.get(ch_idx, f'Ch {ch_idx}')

        flat = ch_data.flatten()
        if len(flat) > 5000:
            flat = np.random.choice(flat, 5000, replace=False)

        ax.hist(flat, bins=30, alpha=0.5, color=color_hex, label=label)

    ax.tick_params(colors='white', labelsize=6)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if len(data_list) <= 4:
        ax.legend(fontsize=6,
                  loc='upper right',
                  facecolor='#1E1E1E',
                  edgecolor='none',
                  labelcolor='white')

    plt.tight_layout(pad=0.2)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)

    img = Image.open(buf)
    return np.array(img)


def get_plane_cache_key(zarr_path: str, h_axis: str, v_axis: str,
                        indices: Dict[str, int], fov_center: Tuple[int, int],
                        zoom_level: int) -> str:
    """Generate cache key."""
    indices_str = "_".join(f"{k}{v}" for k, v in sorted(indices.items()))
    return f"{zarr_path}|{h_axis}{v_axis}|{indices_str}|{fov_center[0]}_{fov_center[1]}|{zoom_level}"


def init_plane_cache():
    """Initialize plane cache."""
    if 'plane_cache' not in st.session_state:
        st.session_state.plane_cache = {}
        st.session_state.plane_cache_order = []


def get_cached_plane(cache_key: str, pyr: Pyramid, metadata: Dict, h_axis: str,
                     v_axis: str, indices: Dict[str, int],
                     fov_center: Tuple[int, int], zoom_level: int):
    """Get plane with LRU caching."""
    init_plane_cache()

    if cache_key in st.session_state.plane_cache:
        st.session_state.plane_cache_order.remove(cache_key)
        st.session_state.plane_cache_order.insert(0, cache_key)
        return st.session_state.plane_cache[cache_key]

    t0 = time.time()
    plane_data, level_used, meta = extract_plane_optimized(
        pyr, metadata, h_axis, v_axis, indices, fov_center, zoom_level)
    t1 = time.time()
    print(f"[EXTRACT] {t1-t0:.3f}s")

    result = (plane_data, level_used, meta)
    st.session_state.plane_cache[cache_key] = result
    st.session_state.plane_cache_order.insert(0, cache_key)

    while len(st.session_state.plane_cache_order) > PLANE_CACHE_SIZE:
        old_key = st.session_state.plane_cache_order.pop()
        st.session_state.plane_cache.pop(old_key, None)

    return result


def init_editor_state():
    """Initialize all state variables."""
    defaults = {
        'vce_h_axis': 'X',
        'vce_v_axis': 'Y',
        'vce_indices': {},
        'vce_fov_center': None,
        'vce_editing_colors': {},
        'vce_editing_labels': {},
        'vce_intensity_limits': {},
        'vce_changes_locked': False,
        'vce_selected_channels': None,
        'vce_intensity_range_mode': "Data Values",
        'vce_zoom_level': 0,
        'vce_current_zarr': None,
        'vce_browser_mode': 'hidden',  # 'hidden', 'breadcrumb', 'active'
        'vce_browse_path': os.path.expanduser("~"),
        'vce_show_histogram': True,
        'vce_dataset_id': 0,
        'vce_cached_scan': None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    init_plane_cache()


def reset_for_new_dataset():
    """Reset state when switching datasets."""
    preserved_keys = {'editor_view_selection', 'operation_mode'}
    preserved_values = {
        k: st.session_state.get(k)
        for k in preserved_keys if k in st.session_state
    }

    keys_to_reset = [
        'vce_h_axis', 'vce_v_axis', 'vce_indices', 'vce_fov_center',
        'vce_editing_colors', 'vce_editing_labels', 'vce_intensity_limits',
        'vce_changes_locked', 'vce_selected_channels', 'vce_zoom_level'
    ]

    for key in keys_to_reset:
        if key in st.session_state:
            if key in [
                    'vce_editing_colors', 'vce_editing_labels',
                    'vce_intensity_limits', 'vce_indices'
            ]:
                st.session_state[key] = {}
            elif key in ['vce_selected_channels', 'vce_fov_center']:
                st.session_state[key] = None
            elif key == 'vce_changes_locked':
                st.session_state[key] = False
            elif key == 'vce_zoom_level':
                st.session_state[key] = 0
            else:
                st.session_state[key] = 'X' if 'h_axis' in key else 'Y'

    st.session_state.plane_cache = {}
    st.session_state.plane_cache_order = []
    st.session_state.vce_dataset_id = st.session_state.get(
        'vce_dataset_id', 0) + 1

    for key in list(st.session_state.keys()):
        if key.startswith('intensity_range_'):
            del st.session_state[key]

    for key, value in preserved_values.items():
        st.session_state[key] = value


def render_channel_controls(num_channels: int, channels_meta: Optional[list],
                            pyr: Pyramid,
                            zarr_path: str) -> Dict[int, Tuple[float, float]]:
    """Render channel editing controls."""
    is_locked = st.session_state.vce_changes_locked
    dataset_id = st.session_state.vce_dataset_id
    use_dtype_range = st.session_state.vce_intensity_range_mode == "Data Type"

    st.markdown("### Channel Settings")

    st.markdown("**Intensity Range**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Actual Data", 
                     help="Scale to min/max values in the data (computed on demand)",
                     use_container_width=True,
                     key=f"vce_range_actual_{dataset_id}",
                     type="primary" if not use_dtype_range else "secondary"):
            st.session_state.vce_intensity_range_mode = "Data Values"
            st.session_state.vce_intensity_limits = {}
            st.rerun()
    with col2:
        if st.button("🔢 Data Type Range",
                     help="Use full range of the data type (e.g., 0-65535 for uint16)",
                     use_container_width=True,
                     key=f"vce_range_dtype_{dataset_id}",
                     type="primary" if use_dtype_range else "secondary"):
            st.session_state.vce_intensity_range_mode = "Data Type"
            st.session_state.vce_intensity_limits = {}
            st.rerun()

    dtype_range = get_dtype_range(
        pyr.layers['0'].dtype) if use_dtype_range else None

    for ch_idx in range(num_channels):
        is_selected = ch_idx in (st.session_state.vce_selected_channels or [])
        color_hex = st.session_state.vce_editing_colors.get(ch_idx, '#FFFFFF')

        col_check, col_label, col_color = st.columns([1, 3, 3])

        with col_check:
            st.markdown(
                f"<div style='background-color:{color_hex};width:20px;height:20px;"
                f"border-radius:3px;border:1px solid #ccc;margin-top:28px;'></div>",
                unsafe_allow_html=True)
            checked = st.checkbox("Show",
                                  value=is_selected,
                                  key=f"ch_active_{dataset_id}_{ch_idx}",
                                  disabled=is_locked,
                                  label_visibility="collapsed")

        if checked and not is_selected:
            if st.session_state.vce_selected_channels is None:
                st.session_state.vce_selected_channels = []
            st.session_state.vce_selected_channels.append(ch_idx)
            st.session_state.vce_selected_channels.sort()
        elif not checked and is_selected:
            st.session_state.vce_selected_channels.remove(ch_idx)

        with col_label:
            label = st.text_input(
                "Label",
                value=st.session_state.vce_editing_labels.get(
                    ch_idx, f'Channel {ch_idx}'),
                key=f"label_{dataset_id}_{ch_idx}",
                disabled=is_locked)
            st.session_state.vce_editing_labels[ch_idx] = label

        with col_color:
            preset_names = list(COLOR_PRESETS.keys())
            current_color = st.session_state.vce_editing_colors.get(
                ch_idx, '#FFFFFF').upper()

            preset_match = next((name
                                 for name, hex_val in COLOR_PRESETS.items()
                                 if hex_val.upper() == current_color), None)
            preset_options = ["Custom"] + preset_names
            preset_idx = preset_options.index(
                preset_match) if preset_match else 0

            color_preset = st.selectbox("Color",
                                        options=preset_options,
                                        index=preset_idx,
                                        key=f"preset_{dataset_id}_{ch_idx}",
                                        disabled=is_locked)

            if color_preset != "Custom":
                st.session_state.vce_editing_colors[ch_idx] = COLOR_PRESETS[
                    color_preset]
            else:
                new_color = st.color_picker("Pick",
                                            value=current_color,
                                            key=f"color_{dataset_id}_{ch_idx}",
                                            disabled=is_locked)
                st.session_state.vce_editing_colors[ch_idx] = new_color

        state_key = f"vce_intensity_{ch_idx}"

        if use_dtype_range and dtype_range:
            ch_min, ch_max = dtype_range
        else:
            ch_min, ch_max = calculate_intensity_ranges_lazy(
                zarr_path, pyr, ch_idx)

        if state_key not in st.session_state.vce_intensity_limits:
            if use_dtype_range and dtype_range:
                # Data Type Range mode: use full dtype range for bounds,
                # but use metadata window start/end for current position if available
                slider_min, slider_max = dtype_range
                if channels_meta and ch_idx < len(channels_meta):
                    window = channels_meta[ch_idx].get('window', {})
                    default_min = window.get('start', slider_min)
                    default_max = window.get('end', slider_max)
                else:
                    default_min, default_max = slider_min, slider_max
            else:
                # Actual Data mode: always use calculated data range for both bounds and position
                default_min, default_max = ch_min, ch_max
            st.session_state.vce_intensity_limits[state_key] = (default_min,
                                                                default_max)

        current = st.session_state.vce_intensity_limits[state_key]

        if ch_max > ch_min:
            if isinstance(ch_min, float) or isinstance(ch_max, float):
                ch_min = float(ch_min)
                ch_max = float(ch_max)
            step = (ch_max - ch_min) / 1000.0
            new_limits = st.slider("Intensity",
                                   min_value=ch_min,
                                   max_value=ch_max,
                                   value=(max(ch_min, float(current[0])),
                                          min(ch_max, float(current[1]))),
                                   step=step,
                                   key=f"intensity_{dataset_id}_{ch_idx}",
                                   disabled=is_locked)
            st.session_state.vce_intensity_limits[state_key] = new_limits

        st.markdown("---")

    return st.session_state.vce_intensity_limits


def render_save_controls(pyr: Pyramid, num_channels: int, channels_meta: list):
    """Render save/lock controls."""
    st.markdown("### Save Changes")
    dataset_id = st.session_state.vce_dataset_id

    is_locked = st.session_state.vce_changes_locked

    if not is_locked:
        st.info("Adjust settings above. Lock when ready.")
        if st.button("Lock Changes",
                     type="primary",
                     use_container_width=True,
                     key=f"vcdataset_ide_lock_{dataset_id}"):
            st.session_state.vce_changes_locked = True
            st.rerun()
    else:
        st.success("Changes locked. Review, then update.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Unlock",
                         use_container_width=True,
                         key=f"vce_unlock_{dataset_id}"):
                st.session_state.vce_changes_locked = False
                st.rerun()
        with col2:
            if st.button("Update Metadata",
                         type="primary",
                         use_container_width=True,
                         key=f"vce_update_{dataset_id}"):
                try:
                    zarr_path = st.session_state.vce_current_zarr

                    # Use Pyramid directly for fast metadata updates (no overhead)
                    pyr_fresh = Pyramid(zarr_path)

                    for ch_idx in range(num_channels):
                        new_color = st.session_state.vce_editing_colors.get(
                            ch_idx, '#FFFFFF')
                        new_label = st.session_state.vce_editing_labels.get(
                            ch_idx, f'Channel {ch_idx}')
                        state_key = f"vce_intensity_{ch_idx}"
                        new_limits = st.session_state.vce_intensity_limits.get(
                            state_key, (0, 1))

                        pyr_fresh.meta.update_channel(ch_idx,
                                                      color=new_color.lstrip('#'),
                                                      label=new_label,
                                                      start_intensity=new_limits[0],
                                                      end_intensity=new_limits[1],
                                                      dtype=pyr.dtype)

                    pyr_fresh.meta.save_changes()

                    load_pyramid_cached.clear()
                    get_pyramid_metadata.clear()

                    st.success("Metadata updated successfully!")
                    st.session_state.vce_changes_locked = False
                    reset_for_new_dataset()
                    st.rerun()

                except Exception as e:
                    st.error(f"Failed to update metadata: {e}")


def is_ome_zarr(path: str) -> bool:
    """Check if a path is an OME-Zarr directory."""
    if not os.path.isdir(path):
        return False
    zattrs_path = os.path.join(path, '.zattrs')
    if not os.path.exists(zattrs_path):
        return False
    try:
        import json
        with open(zattrs_path, 'r') as f:
            attrs = json.load(f)
        return 'multiscales' in attrs
    except Exception:
        return False


def scan_directory_for_zarrs(browse_path: str) -> Dict:
    """
    Scan a directory for contents and OME-Zarr files.
    Returns a dict with folders and zarr files found.
    This is ONLY called when user explicitly requests a scan.
    """
    result = {
        'path': browse_path,
        'folders': [],
        'zarr_files': [],
        'scan_time': time.time()
    }

    if not os.path.exists(browse_path) or not os.path.isdir(browse_path):
        return result

    try:
        items = sorted(os.listdir(browse_path))
        for item in items:
            if item.startswith('.'):
                continue
            item_path = os.path.join(browse_path, item)
            if os.path.isdir(item_path):
                if is_ome_zarr(item_path):
                    result['zarr_files'].append({
                        'name': item,
                        'path': item_path
                    })
                else:
                    result['folders'].append({'name': item, 'path': item_path})
    except Exception as e:
        print(f"[SCAN] Error: {e}")

    return result


def render_sidebar_file_browser():
    """
    Sidebar file browser with CACHED directory scanning and PAGINATION.
    Two modes: 'inactive' (minimal UI) and 'active' (paginated file browser).
    Pagination renders only ~20 items per page for performance with 200+ files.
    Auto-navigates to the page containing the current file when entering browsing mode.
    """
    ITEMS_PER_PAGE = 20
    
    init_editor_state()
    
    if 'vce_current_page' not in st.session_state:
        st.session_state.vce_current_page = 0

    with st.sidebar:
        st.markdown("### OME-Zarr File")

        current_zarr = st.session_state.vce_current_zarr
        browser_mode = st.session_state.vce_browser_mode
        is_browsing = browser_mode == 'active'
        
        cached = st.session_state.vce_cached_scan
        browse_path = st.session_state.vce_browse_path
        
        current_idx = None
        all_zarrs = []
        if cached and cached['path'] == browse_path:
            all_zarrs = cached['zarr_files']
            if current_zarr:
                for i, z in enumerate(all_zarrs):
                    if z['path'] == current_zarr:
                        current_idx = i
                        break

        if current_zarr:
            st.success(f"**Loaded: {Path(current_zarr).name}**")
            if current_idx is not None:
                st.caption(f"📍 Position: {current_idx + 1} of {len(all_zarrs)}")

        if not is_browsing:
            st.markdown("---")
            if st.button("Browse Files",
                         use_container_width=True,
                         key="vce_start_browsing",
                         type="primary"):
                st.session_state.vce_browser_mode = 'active'
                if st.session_state.vce_cached_scan is None:
                    st.session_state.vce_cached_scan = scan_directory_for_zarrs(
                        st.session_state.vce_browse_path)
                    cached = st.session_state.vce_cached_scan
                    all_zarrs = cached['zarr_files'] if cached else []
                if current_idx is not None and all_zarrs:
                    st.session_state.vce_current_page = current_idx // ITEMS_PER_PAGE
                st.rerun()
            return

        # Browser is active - show full browser UI
        st.markdown("---")
        
        if st.button("Done Browsing",
                     use_container_width=True,
                     key="vce_done_browsing"):
            st.session_state.vce_browser_mode = 'inactive'
            st.rerun()

        new_path = st.text_input("Path:",
                                 value=browse_path,
                                 placeholder="Enter path or browse below",
                                 key=f"vce_path_input_{browse_path}")

        if new_path and new_path != browse_path:
            if os.path.exists(new_path):
                if is_ome_zarr(new_path):
                    st.session_state.vce_current_zarr = new_path
                    st.session_state.vce_browse_path = os.path.dirname(new_path)
                    st.session_state.vce_browser_mode = 'inactive'
                    reset_for_new_dataset()
                    st.rerun()
                elif os.path.isdir(new_path):
                    st.session_state.vce_browse_path = new_path
                    st.session_state.vce_cached_scan = scan_directory_for_zarrs(new_path)
                    st.session_state.vce_current_page = 0
                    st.rerun()
                else:
                    st.warning("Not a valid directory or OME-Zarr")
            else:
                st.warning("Path not found")

        st.caption(f"📁 {browse_path}")

        nav_col1, nav_col2 = st.columns(2)
        with nav_col1:
            if st.button("⬆ Parent",
                         key="vce_nav_parent",
                         use_container_width=True):
                parent = os.path.dirname(browse_path)
                if parent and parent != browse_path:
                    st.session_state.vce_browse_path = parent
                    st.session_state.vce_cached_scan = scan_directory_for_zarrs(parent)
                    st.session_state.vce_current_page = 0
                    st.rerun()
        with nav_col2:
            if st.button("🏠 Home",
                         key="vce_nav_home",
                         use_container_width=True):
                home = os.path.expanduser("~")
                st.session_state.vce_browse_path = home
                st.session_state.vce_cached_scan = scan_directory_for_zarrs(home)
                st.session_state.vce_current_page = 0
                st.rerun()

        if cached and cached['path'] == browse_path:
            folders = cached['folders']
            
            if current_idx is not None:
                prev_col, next_col = st.columns(2)
                with prev_col:
                    has_prev = current_idx > 0
                    if st.button("◀ Prev File", 
                                key="vce_nav_prev",
                                use_container_width=True,
                                disabled=not has_prev):
                        prev_zarr = all_zarrs[current_idx - 1]['path']
                        st.session_state.vce_current_zarr = prev_zarr
                        new_idx = current_idx - 1
                        new_page = (len(folders) + new_idx) // ITEMS_PER_PAGE
                        st.session_state.vce_current_page = new_page
                        reset_for_new_dataset()
                        st.rerun()
                with next_col:
                    has_next = current_idx < len(all_zarrs) - 1
                    if st.button("Next File ▶",
                                key="vce_nav_next", 
                                use_container_width=True,
                                disabled=not has_next):
                        next_zarr = all_zarrs[current_idx + 1]['path']
                        st.session_state.vce_current_zarr = next_zarr
                        new_idx = current_idx + 1
                        new_page = (len(folders) + new_idx) // ITEMS_PER_PAGE
                        st.session_state.vce_current_page = new_page
                        reset_for_new_dataset()
                        st.rerun()

            unified_items = []
            for f in folders:
                unified_items.append({'type': 'folder', 'name': f['name'], 'path': f['path']})
            for z in all_zarrs:
                unified_items.append({'type': 'zarr', 'name': z['name'], 'path': z['path']})
            
            if unified_items:
                st.markdown("---")
                search_term = st.text_input("🔍 Filter:", 
                                            placeholder="Type to filter...",
                                            key="vce_file_search")
                
                if search_term:
                    filtered_items = [item for item in unified_items 
                                     if search_term.lower() in item['name'].lower()]
                    st.session_state.vce_current_page = 0
                else:
                    filtered_items = unified_items
                
                total_count = len(unified_items)
                filtered_count = len(filtered_items)
                total_pages = max(1, (filtered_count + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
                
                current_page = st.session_state.vce_current_page
                if current_page >= total_pages:
                    current_page = total_pages - 1
                    st.session_state.vce_current_page = current_page
                
                start_idx = current_page * ITEMS_PER_PAGE
                end_idx = min(start_idx + ITEMS_PER_PAGE, filtered_count)
                page_items = filtered_items[start_idx:end_idx]
                
                folder_count = len(folders)
                zarr_count = len(all_zarrs)
                if search_term:
                    st.caption(f"Showing {filtered_count} of {total_count} items")
                else:
                    st.caption(f"Page {current_page + 1}/{total_pages} ({folder_count} folders, {zarr_count} files)")
                
                pg_col1, pg_col2, pg_col3 = st.columns([1, 2, 1])
                with pg_col1:
                    if st.button("◀", key="vce_page_prev", 
                                use_container_width=True,
                                disabled=current_page <= 0):
                        st.session_state.vce_current_page = current_page - 1
                        st.rerun()
                with pg_col2:
                    if current_idx is not None and not search_term:
                        target_unified_idx = len(folders) + current_idx
                        target_page = target_unified_idx // ITEMS_PER_PAGE
                        if target_page != current_page:
                            if st.button(f"Go to #{current_idx + 1}", 
                                        key="vce_goto_current",
                                        use_container_width=True):
                                st.session_state.vce_current_page = target_page
                                st.rerun()
                        else:
                            st.caption(f"Items {start_idx + 1}-{end_idx}")
                    else:
                        st.caption(f"Items {start_idx + 1}-{end_idx}")
                with pg_col3:
                    if st.button("▶", key="vce_page_next",
                                use_container_width=True,
                                disabled=current_page >= total_pages - 1):
                        st.session_state.vce_current_page = current_page + 1
                        st.rerun()
                
                name_style = "white-space: normal; overflow-wrap: anywhere; word-break: break-word; line-height: 1.3;"
                
                for item in page_items:
                    if item['type'] == 'folder':
                        folder_idx = folders.index({'name': item['name'], 'path': item['path']})
                        col_name, col_btn = st.columns([3, 1])
                        with col_name:
                            full_name = item['name']
                            full_path = item['path']
                            st.markdown(
                                f'<div style="{name_style}" title="{full_path}"><b>📁 {full_name}</b></div>',
                                unsafe_allow_html=True
                            )
                        with col_btn:
                            if st.button("Enter",
                                         key=f"vce_select_folder_{folder_idx}"):
                                st.session_state.vce_browse_path = item['path']
                                st.session_state.vce_cached_scan = scan_directory_for_zarrs(
                                    item['path'])
                                st.session_state.vce_current_page = 0
                                st.rerun()
                    else:
                        zarr_idx = next((i for i, z in enumerate(all_zarrs) 
                                        if z['path'] == item['path']), 0)
                        col_name, col_btn = st.columns([3, 1])
                        with col_name:
                            is_current = current_zarr == item['path']
                            prefix = "✓ " if is_current else ""
                            full_name = item['name']
                            full_path = item['path']
                            st.markdown(
                                f'<div style="{name_style}" title="{full_path}">{prefix}🔬 {full_name}</div>',
                                unsafe_allow_html=True
                            )
                        with col_btn:
                            if st.button("Open",
                                         key=f"vce_open_zarr_{zarr_idx}",
                                         disabled=is_current):
                                st.session_state.vce_current_zarr = item['path']
                                reset_for_new_dataset()
                                st.rerun()

            if not unified_items:
                st.info("No OME-Zarr files or folders found")
        else:
            if st.button("Scan Directory", key="vce_initial_scan"):
                st.session_state.vce_cached_scan = scan_directory_for_zarrs(browse_path)
                st.session_state.vce_current_page = 0
                st.rerun()


def render(bridge=None):
    """Render the Visual Channel Editor interface."""
    st.subheader("Visual Channel Editor")

    init_editor_state()

    zarr_path = st.session_state.vce_current_zarr

    if not zarr_path:
        st.info("Enter an OME-Zarr path in the sidebar to start editing.")
        return

    zarr_path_obj = Path(zarr_path)
    if not zarr_path_obj.exists():
        st.warning("Selected path does not exist.")
        st.session_state.vce_current_zarr = None
        return

    pyr = load_pyramid_cached(zarr_path)
    if pyr is None:
        return

    version_warnings = st.session_state.get('vce_version_warning', {})
    if zarr_path in version_warnings:
        detected_version = version_warnings[zarr_path]
        st.warning(
            f"This OME-Zarr lacks a version field and is therefore invalid. "
            f"Version {detected_version} was determined based on the metadata structure."
        )

    metadata = get_pyramid_metadata(zarr_path)
    axes = metadata['axes']
    shape = metadata['shape']

    channels_meta = getattr(pyr.meta, 'channels', None)

    if not channels_meta:
        c_idx = axes.find('c')
        if c_idx >= 0:
            n_channels = shape[c_idx]
        else:
            n_channels = 1
        
        dtype = pyr.dtype
        pyr.meta.autocompute_omerometa(n_channels, dtype)
        channels_meta = pyr.meta.channels
        
        st.warning(
            "This OME-Zarr does not have channel metadata (omero). "
            "The view was created with default channel metadata."
        )

    num_channels = len(channels_meta)

    if not st.session_state.vce_editing_colors:
        for ch_idx in range(num_channels):
            if channels_meta and ch_idx < len(channels_meta):
                color = channels_meta[ch_idx].get('color', 'FFFFFF')
                if not color.startswith('#'):
                    color = '#' + color
            else:
                default_colors = [
                    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF',
                    '#00FFFF'
                ]
                color = default_colors[ch_idx % len(default_colors)]
            st.session_state.vce_editing_colors[ch_idx] = color

    if not st.session_state.vce_editing_labels:
        for ch_idx in range(num_channels):
            if channels_meta and ch_idx < len(channels_meta):
                label = channels_meta[ch_idx].get('label', f'Channel {ch_idx}')
            else:
                label = f'Channel {ch_idx}'
            st.session_state.vce_editing_labels[ch_idx] = label

    if st.session_state.vce_selected_channels is None:
        st.session_state.vce_selected_channels = list(
            range(min(num_channels, 3)))

    spatial_axes = [a.upper() for a in axes if a in 'xyz']
    h_axis = st.session_state.vce_h_axis
    if h_axis not in spatial_axes or (h_axis != 'X' and 'X' in spatial_axes):
        h_axis = 'X' if 'X' in spatial_axes else (spatial_axes[-1] if spatial_axes else 'X')
        st.session_state.vce_h_axis = h_axis

    available_v_axes = [a for a in spatial_axes if a != h_axis]
    v_axis = st.session_state.vce_v_axis
    if v_axis not in available_v_axes or (v_axis != 'Y' and 'Y' in available_v_axes):
        if 'Y' in available_v_axes:
            v_axis = 'Y'
        elif available_v_axes:
            v_axis = available_v_axes[0]
        else:
            v_axis = 'Y'
        st.session_state.vce_v_axis = v_axis

    h_idx = axes.find(h_axis.lower())
    v_idx = axes.find(v_axis.lower())
    plane_width = shape[h_idx] if h_idx >= 0 else shape[-1]
    plane_height = shape[v_idx] if v_idx >= 0 else shape[-2]

    if st.session_state.vce_fov_center is None:
        st.session_state.vce_fov_center = (plane_height // 2, plane_width // 2)

    num_levels = metadata['num_levels']
    if st.session_state.vce_zoom_level >= num_levels:
        st.session_state.vce_zoom_level = max(0, num_levels - 1)

    st.markdown(f"**{zarr_path_obj.name}** — {axes.upper()} {shape}")

    left_col, right_col = st.columns([1, 2])

    with left_col:
        with st.container(border=True):
            render_channel_controls(num_channels, channels_meta, pyr,
                                    zarr_path)
            st.markdown("---")
            render_save_controls(pyr, num_channels, channels_meta)

    with right_col:
        cache_key = get_plane_cache_key(zarr_path, h_axis, v_axis,
                                        st.session_state.vce_indices,
                                        st.session_state.vce_fov_center,
                                        st.session_state.vce_zoom_level)

        plane_data, level_used, meta = get_cached_plane(
            cache_key, pyr, metadata, h_axis, v_axis,
            st.session_state.vce_indices, st.session_state.vce_fov_center,
            st.session_state.vce_zoom_level)

        selected_channels = st.session_state.vce_selected_channels or list(
            range(num_channels))

        if plane_data.ndim == 3 and plane_data.shape[0] > 1:
            num_extracted = plane_data.shape[0]
            valid_indices = [
                idx for idx in selected_channels if idx < num_extracted
            ]
            if valid_indices:
                plane_data = plane_data[valid_indices]
                channel_indices = valid_indices
            else:
                channel_indices = [0]
        else:
            channel_indices = selected_channels[:1] if selected_channels else [
                0
            ]

        intensity_limits = {}
        for ch_idx in channel_indices:
            state_key = f"vce_intensity_{ch_idx}"
            if state_key in st.session_state.vce_intensity_limits:
                intensity_limits[
                    ch_idx] = st.session_state.vce_intensity_limits[state_key]
            else:
                ch_min, ch_max = calculate_intensity_ranges_lazy(
                    zarr_path, pyr, ch_idx)
                intensity_limits[ch_idx] = (ch_min, ch_max)

        rgb_image = normalize_for_display_fast(
            plane_data, channel_indices, intensity_limits,
            st.session_state.vce_editing_colors)

        rgb_canvas = fit_to_canvas(rgb_image, CANVAS_SIZE)

        dataset_id = st.session_state.vce_dataset_id

        with st.container(height=550, border=True):
            img_col, hist_col = st.columns([2, 1])

            with img_col:
                st.image(rgb_canvas, width=350)
                st.caption(f"Level {level_used} | Scale {meta['scale_factor']:.1f}x")

            with hist_col:
                show_hist = st.checkbox("Histogram",
                                        value=st.session_state.vce_show_histogram,
                                        key=f"vce_hist_{dataset_id}")
                st.session_state.vce_show_histogram = show_hist

                if show_hist:
                    hist_img = render_histogram_fast(
                        plane_data, channel_indices,
                        st.session_state.vce_editing_colors,
                        st.session_state.vce_editing_labels)
                    st.image(hist_img, width=200)

            nav_col1, nav_col2, nav_col3 = st.columns(3)

            with nav_col1:
                if num_levels > 1:
                    new_zoom = st.slider("Zoom Level",
                                         min_value=0,
                                         max_value=num_levels - 1,
                                         value=st.session_state.vce_zoom_level,
                                         key=f"vce_zoom_{dataset_id}")
                    if new_zoom != st.session_state.vce_zoom_level:
                        st.session_state.vce_zoom_level = new_zoom
                        st.rerun()

            with nav_col2:
                t_idx = axes.find('t')
                if t_idx >= 0:
                    t_max = shape[t_idx] - 1
                    if t_max > 0:
                        current_t = st.session_state.vce_indices.get('t', 0)
                        new_t = st.slider("Time",
                                          0,
                                          t_max,
                                          current_t,
                                          key=f"vce_time_{dataset_id}")
                        if new_t != current_t:
                            st.session_state.vce_indices['t'] = new_t
                            st.rerun()
                    else:
                        st.caption("Time: 0 (single frame)")

            with nav_col3:
                z_idx = axes.find('z')
                if z_idx >= 0:
                    z_max = shape[z_idx] - 1
                    if z_max > 0:
                        current_z = st.session_state.vce_indices.get('z', 0)
                        new_z = st.slider("Z Slice",
                                          0,
                                          z_max,
                                          current_z,
                                          key=f"vce_z_{dataset_id}")
                        if new_z != current_z:
                            st.session_state.vce_indices['z'] = new_z
                            st.rerun()
                    else:
                        st.caption("Z: 0 (single slice)")

            with st.expander("Position"):
                center_row, center_col = st.session_state.vce_fov_center

                h_max = plane_width - 1
                v_max = plane_height - 1
                h_min = 0
                v_min = 0

                new_col = center_col
                new_row = center_row

                if h_max > h_min:
                    new_col = st.slider(f"{h_axis} Position",
                                        h_min,
                                        h_max,
                                        min(max(center_col, h_min), h_max),
                                        key=f"vce_hpos_{dataset_id}")
                else:
                    st.caption(f"{h_axis}: 0 (single pixel)")

                if v_max > v_min:
                    new_row = st.slider(f"{v_axis} Position",
                                        v_min,
                                        v_max,
                                        min(max(center_row, v_min), v_max),
                                        key=f"vce_vpos_{dataset_id}")
                else:
                    st.caption(f"{v_axis}: 0 (single pixel)")

                if new_col != center_col or new_row != center_row:
                    st.session_state.vce_fov_center = (new_row, new_col)
                    st.rerun()
