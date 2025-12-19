"""Pixel Metadata Update View for EuBI-Bridge GUI"""
import streamlit as st
import os
import pandas as pd
from eubi_bridge.views.shared import run_operation_with_logging

SPACE_UNITS = [
    "(keep unchanged)",
    "angstrom",
    "centimeter",
    "decimeter",
    "meter",
    "micrometer",
    "millimeter",
    "nanometer",
    "picometer",
]

TIME_UNITS = [
    "(keep unchanged)",
    "attosecond",
    "day",
    "femtosecond",
    "hour",
    "microsecond",
    "millisecond",
    "minute",
    "nanosecond",
    "picosecond",
    "second",
]


def load_pyramid(path):
    """
    Load and validate an OME-Zarr using the Pyramid class.
    Returns (pyramid, error_message) tuple.
    Uses the cached loader from visual_channel_editor for performance.
    """
    if not path or not os.path.exists(path):
        return None, "Path does not exist"
    
    try:
        from views.visual_channel_editor import load_pyramid_cached
        pyr = load_pyramid_cached(path)
        if pyr is None:
            return None, "Failed to load OME-Zarr"
        return pyr, None
    except ValueError as e:
        return None, f"Not a valid OME-Zarr: {str(e)}"
    except Exception as e:
        return None, f"Error loading OME-Zarr: {str(e)}"


def format_bytes(size_bytes):
    """Format bytes into human-readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def get_storage_info(layer, zarr_path=None, level_path=None):
    """Extract chunk, shard, compression info and sizes from a zarr array."""
    chunks = None
    shards = None
    compressor = None
    chunk_memory_size = None
    chunk_disk_size = None
    
    try:
        if hasattr(layer, 'chunks'):
            chunks = layer.chunks
        
        if chunks and hasattr(layer, 'dtype'):
            import numpy as np
            chunk_elements = np.prod(chunks)
            itemsize = np.dtype(layer.dtype).itemsize
            chunk_memory_size = int(chunk_elements * itemsize)
        
        if hasattr(layer, 'compressor') and layer.compressor is not None:
            comp = layer.compressor
            comp_name = type(comp).__name__
            if hasattr(comp, 'cname'):
                compressor = f"{comp_name} ({comp.cname})"
            elif hasattr(comp, 'codec_id'):
                compressor = f"{comp_name} ({comp.codec_id})"
            else:
                compressor = comp_name
        
        if hasattr(layer, 'metadata'):
            meta = layer.metadata
            if hasattr(meta, 'codecs'):
                for codec in meta.codecs:
                    codec_name = type(codec).__name__
                    if 'shard' in codec_name.lower():
                        if hasattr(codec, 'chunk_shape'):
                            shards = chunks
                            chunks = codec.chunk_shape
                            if chunks and hasattr(layer, 'dtype'):
                                import numpy as np
                                chunk_elements = np.prod(chunks)
                                itemsize = np.dtype(layer.dtype).itemsize
                                chunk_memory_size = int(chunk_elements * itemsize)
                        break
                for codec in meta.codecs:
                    codec_name = type(codec).__name__
                    if codec_name.lower() in ['blosc', 'gzip', 'zstd', 'lz4', 'zlib']:
                        compressor = codec_name
                        break
                    if hasattr(codec, 'cname'):
                        compressor = f"{codec_name} ({codec.cname})"
                        break
        
        if zarr_path and level_path and chunks:
            import os
            level_dir = os.path.join(zarr_path, level_path)
            if os.path.isdir(level_dir):
                total_size = 0
                chunk_count = 0
                for root, dirs, files in os.walk(level_dir):
                    for f in files:
                        if not f.startswith('.') and f not in ['zarray', 'zattrs', '.zarray', '.zattrs', 'zarr.json']:
                            fp = os.path.join(root, f)
                            try:
                                total_size += os.path.getsize(fp)
                                chunk_count += 1
                            except OSError:
                                pass
                if chunk_count > 0:
                    chunk_disk_size = total_size // chunk_count
    except Exception:
        pass
    
    return chunks, shards, compressor, chunk_memory_size, chunk_disk_size


def display_ome_zarr_info(pyr, title="OME-Zarr Information"):
    """Display comprehensive information about an OME-Zarr using Pyramid."""
    st.subheader(title)
    
    try:
        name = pyr.meta.tag or 'Unnamed'
    except (KeyError, AttributeError):
        name = 'Unnamed'
    
    zarr_path = str(pyr.meta.store.path) if hasattr(pyr.meta, 'store') and hasattr(pyr.meta.store, 'path') else None
    if zarr_path is None and hasattr(pyr, 'path'):
        zarr_path = str(pyr.path)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"**Name:** {name}")
    with col2:
        st.markdown(f"**NGFF Version:** {pyr.meta.version}")
    with col3:
        st.markdown(f"**Resolution Levels:** {pyr.nlayers}")
    with col4:
        st.markdown(f"**Data Type:** {pyr.dtype}")
    
    col5, col6 = st.columns(2)
    with col5:
        st.markdown(f"**Axes:** {pyr.axes.upper()}")
    with col6:
        base_layer = pyr.layers.get('0') or pyr.layers.get(pyr.meta.resolution_paths[0]) if pyr.meta.resolution_paths else None
        if base_layer is not None:
            _, _, compressor, _, _ = get_storage_info(base_layer)
            if compressor:
                st.markdown(f"**Compression:** {compressor}")
    
    st.markdown("**Pyramid Layers:**")
    
    axis_order = pyr.axes
    unit_dict = pyr.meta.unit_dict
    axes_metadata = pyr.meta.multiscales.get('axes', [])
    resolution_paths = pyr.meta.resolution_paths
    
    for level_idx, path in enumerate(resolution_paths):
        level_scales = pyr.meta.scaledict.get(path, {})
        layer = pyr.layers.get(path)
        layer_shape = layer.shape if layer is not None else "N/A"
        
        chunks, shards, compressor, chunk_mem, chunk_disk = get_storage_info(layer, zarr_path, path) if layer is not None else (None, None, None, None, None)
        
        level_label = f"Level {path} — Shape: {layer_shape}"
        if chunks:
            size_info = ""
            if chunk_mem:
                size_info += f" ~{format_bytes(chunk_mem)}"
            if chunk_disk:
                size_info += f" / {format_bytes(chunk_disk)} on disk"
            level_label += f" | Chunks: {chunks}{size_info}"
        if shards:
            level_label += f" | Shards: {shards}"
        
        expanded = (level_idx == 0)
        with st.expander(level_label, expanded=expanded):
            axes_data = []
            for i, ax in enumerate(axis_order):
                ax_meta = axes_metadata[i] if i < len(axes_metadata) else {}
                ax_type = ax_meta.get('type', 'unknown')
                ax_unit = unit_dict.get(ax, None)
                ax_scale = level_scales.get(ax, 1.0)
                
                axes_data.append({
                    "Axis": ax.upper(),
                    "Type": ax_type,
                    "Scale": f"{ax_scale:.4g}",
                    "Unit": ax_unit if ax_unit else "(none)"
                })
            
            df = pd.DataFrame(axes_data)
            st.dataframe(df, hide_index=True, use_container_width=True)
    
    if hasattr(pyr.meta, 'channels') and pyr.meta.channels:
        st.markdown("**Channel Summary:**")
        channels = pyr.meta.channels
        channel_labels = []
        for i, ch in enumerate(channels):
            label = ch.get('label')
            if label is None or label == '':
                label = f'Channel {i}'
            channel_labels.append(label)
        st.markdown(f"{len(channels)} channels: {', '.join(channel_labels)}")


def get_editable_axes_info(pyr):
    """Get axes information for editing (excluding channel axes)."""
    axis_order = pyr.axes
    unit_dict = pyr.meta.unit_dict
    base_scales = pyr.meta.scaledict.get('0', {})
    axes_metadata = pyr.meta.multiscales.get('axes', [])
    
    editable_axes = []
    for i, ax in enumerate(axis_order):
        ax_meta = axes_metadata[i] if i < len(axes_metadata) else {}
        ax_type = ax_meta.get('type', 'unknown')
        
        if ax_type != 'channel':
            editable_axes.append({
                'name': ax,
                'type': ax_type,
                'unit': unit_dict.get(ax, None),
                'scale': base_scales.get(ax, 1.0),
            })
    
    return editable_axes


def render(bridge):
    """Render the pixel metadata update UI"""
    st.subheader("Inspect/Edit Pixel Metadata")
    
    input_path = st.session_state.get('vce_current_zarr', None)
    
    if not input_path:
        st.info("Select an OME-Zarr file from the sidebar to start editing.")
        return
    
    if 'pixel_meta_pyramid' not in st.session_state:
        st.session_state.pixel_meta_pyramid = None
        st.session_state.pixel_meta_path = None

    if st.session_state.pixel_meta_path != input_path:
        st.session_state.pixel_meta_pyramid = None
        st.session_state.pixel_meta_path = input_path

    pyr, error = load_pyramid(input_path)
    
    if pyr:
        st.session_state.pixel_meta_pyramid = pyr
    else:
        st.error(error)
        st.session_state.pixel_meta_pyramid = None
        return

    pyr = st.session_state.pixel_meta_pyramid
    
    if pyr:
        with st.container(height=550, border=True):
            st.markdown("---")
            display_ome_zarr_info(pyr, title="Current OME-Zarr Information")
            
            editable_axes = get_editable_axes_info(pyr)
            
            if not editable_axes:
                st.info("No editable axes found (only channel axes present)")
                return
            
            st.markdown("---")
            st.subheader("Update Scale and Unit Values")
            st.markdown("Leave scale empty to keep unchanged. Select '(keep unchanged)' for units to preserve current value.")
            
            new_scales = {}
            new_units = {}
            
            for axis in editable_axes:
                axis_name = axis['name']
                axis_type = axis['type']
                
                st.markdown(f"**{axis_name.upper()} axis** ({axis_type})")
                
                col_scale, col_unit = st.columns(2)
                
                with col_scale:
                    new_scale = st.text_input(
                        f"New scale for {axis_name.upper()}",
                        value="",
                        placeholder=str(axis['scale']),
                        key=f"pixel_scale_{axis_name}",
                        help=f"Enter new scale value (current: {axis['scale']})"
                    )
                    new_scales[axis_name] = new_scale
                
                with col_unit:
                    if axis_type == "time":
                        unit_options = TIME_UNITS
                    elif axis_type == "space":
                        unit_options = SPACE_UNITS
                    else:
                        unit_options = ["(keep unchanged)"] + SPACE_UNITS[1:] + TIME_UNITS[1:]
                    
                    current_unit = axis['unit'] if axis['unit'] else "(none)"
                    
                    new_unit = st.selectbox(
                        f"New unit for {axis_name.upper()}",
                        options=unit_options,
                        key=f"pixel_unit_{axis_name}",
                        help=f"Select new unit (current: {current_unit})"
                    )
                    new_units[axis_name] = new_unit
            
            st.markdown("---")
            st.subheader("Summary of Changes")
            
            changes = []
            for axis in editable_axes:
                axis_name = axis['name']
                scale_change = new_scales.get(axis_name, "")
                unit_change = new_units.get(axis_name, "(keep unchanged)")
                
                if scale_change or unit_change != "(keep unchanged)":
                    change_parts = []
                    if scale_change:
                        change_parts.append(f"scale: {axis['scale']} → {scale_change}")
                    if unit_change != "(keep unchanged)":
                        old_unit = axis['unit'] if axis['unit'] else "(none)"
                        change_parts.append(f"unit: {old_unit} → {unit_change}")
                    changes.append(f"**{axis_name.upper()}**: {', '.join(change_parts)}")
            
            if changes:
                for change in changes:
                    st.markdown(f"- {change}")
            else:
                st.info("No changes specified")
            
            st.markdown("---")
            
            if st.button("Update Pixel Metadata", type="primary", use_container_width=True):
                if not changes:
                    st.warning("No changes to apply")
                else:
                    st.markdown("### Update Log")
                    status_container = st.empty()
                    log_container = st.container()
                    status_container.info("Updating pixel metadata...")

                    def parse_scale(val):
                        if not val or not val.strip():
                            return None
                        try:
                            return float(val)
                        except:
                            return None

                    def get_unit_value(unit_str):
                        if unit_str == "(keep unchanged)":
                            return None
                        return unit_str

                    time_scale = None
                    z_scale = None
                    y_scale = None
                    x_scale = None
                    time_unit = None
                    z_unit = None
                    y_unit = None
                    x_unit = None

                    for axis in editable_axes:
                        name = axis['name'].lower()
                        scale_val = parse_scale(new_scales.get(axis['name'], ""))
                        unit_val = get_unit_value(new_units.get(axis['name'], "(keep unchanged)"))
                        
                        if name == 't':
                            time_scale = scale_val
                            time_unit = unit_val
                        elif name == 'z':
                            z_scale = scale_val
                            z_unit = unit_val
                        elif name == 'y':
                            y_scale = scale_val
                            y_unit = unit_val
                        elif name == 'x':
                            x_scale = scale_val
                            x_unit = unit_val

                    def run_update():
                        from eubi_bridge.ebridge import EuBIBridge
                        update_bridge = EuBIBridge()
                        update_bridge.update_pixel_meta(
                            input_path=input_path,
                            includes=None,
                            excludes=None,
                            time_scale=time_scale,
                            z_scale=z_scale,
                            y_scale=y_scale,
                            x_scale=x_scale,
                            time_unit=time_unit,
                            z_unit=z_unit,
                            y_unit=y_unit,
                            x_unit=x_unit,
                        )

                    result = run_operation_with_logging(run_update, status_container, log_container)

                    if result['success']:
                        status_container.success("Pixel metadata updated successfully!")
                        st.session_state.pixel_meta_pyramid = None
                        
                        st.markdown("---")
                        updated_pyr, _ = load_pyramid(input_path)
                        if updated_pyr:
                            display_ome_zarr_info(updated_pyr, title="Updated OME-Zarr Information")
                    else:
                        status_container.error(f"Update failed: {result['error']}")
                        if result['traceback']:
                            with st.expander("Show error details"):
                                st.code(result['traceback'])
