"""Channel Metadata Update View for EuBI-Bridge GUI"""
import streamlit as st
import os
import pandas as pd
from eubi_bridge.views.shared import render_path_input, run_operation_with_logging

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


def load_pyramid(path):
    """
    Load and validate an OME-Zarr using the Pyramid class.
    Returns (pyramid, error_message) tuple.
    """
    if not path or not os.path.exists(path):
        return None, "Path does not exist"
    
    try:
        from eubi_bridge.ngff.multiscales import Pyramid
        pyr = Pyramid(path)
        
        if not hasattr(pyr.meta, 'channels') or not pyr.meta.channels:
            return None, "OME-Zarr is missing channel (omero) metadata"
        
        return pyr, None
    except ValueError as e:
        return None, f"Not a valid OME-Zarr: {str(e)}"
    except Exception as e:
        return None, f"Error loading OME-Zarr: {str(e)}"


def display_ome_zarr_info(pyr, title="OME-Zarr Information"):
    """Display comprehensive information about an OME-Zarr using Pyramid."""
    st.subheader(title)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Name:** {pyr.meta.tag or 'Unnamed'}")
    with col2:
        st.markdown(f"**NGFF Version:** {pyr.meta.version}")
    with col3:
        st.markdown(f"**Resolution Levels:** {pyr.nlayers}")
    
    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown(f"**Data Type:** {pyr.dtype}")
    with col5:
        st.markdown(f"**Base Shape:** {pyr.shape}")
    with col6:
        st.markdown(f"**Axes:** {pyr.axes.upper()}")
    
    st.markdown("**Pyramid Layers:**")
    
    axis_order = pyr.axes
    unit_dict = pyr.meta.unit_dict
    axes_metadata = pyr.meta.multiscales.get('axes', [])
    resolution_paths = pyr.meta.resolution_paths
    
    for level_idx, path in enumerate(resolution_paths):
        level_scales = pyr.meta.scaledict.get(path, {})
        layer = pyr.layers.get(path)
        layer_shape = layer.shape if layer is not None else "N/A"
        
        expanded = (level_idx == 0)
        with st.expander(f"Level {path} - Shape: {layer_shape}", expanded=expanded):
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


def display_channel_info(pyr, title="Channel Information"):
    """Display channel information with color swatches."""
    st.markdown(f"**{title}:**")
    
    channels = pyr.meta.channels
    num_channels = len(channels)
    
    if num_channels <= 6:
        cols = st.columns(num_channels)
        for i, ch in enumerate(channels):
            with cols[i]:
                label = ch.get('label', f'Channel {i}')
                color = ch.get('color', 'FFFFFF')
                if not color.startswith('#'):
                    color = '#' + color
                
                text_color = 'black' if color.upper() in ['#FFFFFF', '#FFFF00', '#00FFFF', '#FFC0CB', '#00FF00'] else 'white'
                st.markdown(f"""
                <div style="background-color: {color}; padding: 8px; border-radius: 5px; text-align: center; color: {text_color}; border: 1px solid #ccc;">
                    <strong>Ch {i}</strong><br>
                    {label}
                </div>
                <div style="text-align: center; font-size: 12px; color: #666; margin-top: 4px;">{color}</div>
                """, unsafe_allow_html=True)
    else:
        for i, ch in enumerate(channels):
            label = ch.get('label', f'Channel {i}')
            color = ch.get('color', 'FFFFFF')
            if not color.startswith('#'):
                color = '#' + color
            
            col1, col2 = st.columns([1, 5])
            with col1:
                st.markdown(f"""
                <div style="background-color: {color}; width: 40px; height: 24px; border-radius: 3px; border: 1px solid #ccc;"></div>
                <div style="font-size: 10px; color: #666;">{color}</div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"**Ch {i}:** {label}")


def get_channel_info(pyr):
    """Extract channel information from Pyramid."""
    channels = pyr.meta.channels
    
    channel_info = []
    for i, ch in enumerate(channels):
        color = ch.get("color", "FFFFFF")
        if not color.startswith("#"):
            color = "#" + color
        
        info = {
            "index": i,
            "label": ch.get("label", f"Channel {i}"),
            "color": color,
            "window": ch.get("window", {}),
        }
        channel_info.append(info)
    
    return channel_info


def render(bridge):
    """Render the channel metadata update UI"""
    st.header("Edit Channel Metadata")
    st.markdown("View and update channel labels, colors, and intensity limits for OME-Zarr files")

    st.subheader("OME-Zarr Path")
    input_path = render_path_input(
        "channel_meta",
        label="OME-Zarr Path",
        help_text="Path to a single OME-Zarr file"
    )

    if 'channel_meta_pyramid' not in st.session_state:
        st.session_state.channel_meta_pyramid = None
        st.session_state.channel_meta_path = None

    if input_path:
        if st.session_state.channel_meta_path != input_path:
            st.session_state.channel_meta_pyramid = None
            st.session_state.channel_meta_path = input_path

        pyr, error = load_pyramid(input_path)
        
        if pyr:
            num_channels = len(pyr.meta.channels)
            st.success(f"Valid OME-Zarr with {num_channels} channels")
            st.session_state.channel_meta_pyramid = pyr
        else:
            st.error(error)
            st.session_state.channel_meta_pyramid = None

    pyr = st.session_state.channel_meta_pyramid
    
    if pyr:
        st.markdown("---")
        display_ome_zarr_info(pyr, title="Current OME-Zarr Information")
        display_channel_info(pyr, title="Current Channel Configuration")
        
        channel_info = get_channel_info(pyr)
        
        st.markdown("---")
        st.subheader("Select Channels to Update")
        
        selected_channels = []
        num_channels = len(channel_info)
        
        if num_channels <= 6:
            select_cols = st.columns(num_channels)
            for i, ch in enumerate(channel_info):
                with select_cols[i]:
                    if st.checkbox(f"Ch {ch['index']}: {ch['label']}", key=f"select_ch_{i}"):
                        selected_channels.append(i)
        else:
            col1, col2 = st.columns(2)
            for i, ch in enumerate(channel_info):
                with col1 if i % 2 == 0 else col2:
                    if st.checkbox(f"Ch {ch['index']}: {ch['label']}", key=f"select_ch_{i}"):
                        selected_channels.append(i)
        
        if selected_channels:
            st.markdown("---")
            st.subheader("New Metadata for Selected Channels")
            
            new_labels = {}
            new_colors = {}
            
            for ch_idx in selected_channels:
                ch = channel_info[ch_idx]
                st.markdown(f"**Channel {ch_idx}: {ch['label']}**")
                
                col_label, col_color_preset, col_color_hex = st.columns([2, 1, 1])
                
                with col_label:
                    new_label = st.text_input(
                        "New Label",
                        value=ch['label'],
                        key=f"new_label_{ch_idx}",
                        help="Enter a new label for this channel"
                    )
                    new_labels[ch_idx] = new_label
                
                with col_color_preset:
                    color_preset = st.selectbox(
                        "Color Preset",
                        ["(Custom)"] + list(COLOR_PRESETS.keys()),
                        key=f"color_preset_{ch_idx}",
                        help="Choose a preset color or select Custom to enter hex"
                    )
                
                with col_color_hex:
                    if color_preset == "(Custom)":
                        new_color = st.text_input(
                            "Hex Color",
                            value=ch['color'],
                            key=f"new_color_custom_{ch_idx}",
                            help="Enter hex color (e.g., #FF0000)"
                        )
                    else:
                        new_color = COLOR_PRESETS[color_preset]
                        st.text_input(
                            "Hex Color",
                            value=new_color,
                            key=f"new_color_preset_{ch_idx}_{color_preset}",
                            disabled=True,
                            help=f"Color set by preset: {color_preset}"
                        )
                    new_colors[ch_idx] = new_color
                
                preview_color = new_colors[ch_idx]
                if not preview_color.startswith('#'):
                    preview_color = '#' + preview_color
                text_color = 'black' if preview_color.upper() in ['#FFFFFF', '#FFFF00', '#00FFFF', '#FFC0CB', '#00FF00'] else 'white'
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 10px; margin-top: 5px;">
                    <div style="background-color: {preview_color}; padding: 6px 12px; border-radius: 4px; color: {text_color};">
                        <strong>{new_labels[ch_idx]}</strong>
                    </div>
                    <span style="color: #666;">({preview_color})</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            intensity_limits = st.selectbox(
                "Intensity Limits",
                ["from_dtype", "from_data", "auto"],
                help="How to determine intensity limits for channels",
                key="channel_intensity_limits"
            )
            
            st.markdown("---")
            st.subheader("Summary of Changes")
            
            for ch_idx in selected_channels:
                ch = channel_info[ch_idx]
                st.markdown(f"- **Ch {ch_idx}:** {ch['label']} → {new_labels[ch_idx]}, Color: {ch['color']} → {new_colors[ch_idx]}")
            st.markdown(f"- **Intensity limits:** {intensity_limits}")
            
            st.markdown("---")
            
            if st.button("Update Channel Metadata", type="primary", use_container_width=True):
                st.markdown("### Update Log")
                status_container = st.empty()
                log_container = st.container()
                status_container.info("Updating channel metadata...")

                channel_indices_list = selected_channels
                channel_labels_list = [new_labels[i] for i in selected_channels]
                channel_colors_list = [new_colors[i].lstrip('#') for i in selected_channels]

                def run_update():
                    from eubi_bridge.ebridge import EuBIBridge
                    update_bridge = EuBIBridge()
                    update_bridge.update_channel_meta(
                        input_path=input_path,
                        includes=None,
                        excludes=None,
                        channel_indices=channel_indices_list,
                        channel_labels=channel_labels_list,
                        channel_colors=channel_colors_list,
                        channel_intensity_limits=intensity_limits,
                    )

                result = run_operation_with_logging(run_update, status_container, log_container)

                if result['success']:
                    status_container.success("Channel metadata updated successfully!")
                    st.session_state.channel_meta_pyramid = None
                    
                    st.markdown("---")
                    updated_pyr, _ = load_pyramid(input_path)
                    if updated_pyr:
                        display_ome_zarr_info(updated_pyr, title="Updated OME-Zarr Information")
                        display_channel_info(updated_pyr, title="Updated Channel Configuration")
                else:
                    status_container.error(f"Update failed: {result['error']}")
                    if result['traceback']:
                        with st.expander("Show error details"):
                            st.code(result['traceback'])
        else:
            st.info("Select one or more channels above to configure their new metadata")
    
    elif input_path:
        st.info("Please provide a valid OME-Zarr path to continue")
    else:
        st.info("Enter an OME-Zarr path above to begin")
