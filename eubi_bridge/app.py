import streamlit as st
import streamlit.components.v1 as components
import json
import os
import logging
import time
import threading
import queue
import multiprocessing
from pathlib import Path
import sys
import html as html_module
from eubi_bridge.views import pixel_metadata, channel_metadata, lazy_viewer, visual_channel_editor
from eubi_bridge.views.compression_config import render_compression_config

# Fix fork-after-threads deadlock: use spawn context for ProcessPoolExecutor
# This must be set before any multiprocessing usage
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Already set

from eubi_bridge.mp_logging_setup import setup_mp_logging


class QueueHandler(logging.Handler):
    """Logging handler that sends records to a queue for real-time display"""

    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'))

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_queue.put(msg)
        except Exception:
            pass


try:
    from eubi_bridge.ebridge import EuBIBridge
    from eubi_bridge.utils.logging_config import setup_logging
    setup_logging()
except ImportError:
    st.error("EuBI-Bridge library not found. Please ensure the library is properly installed.")
    st.stop()

st.set_page_config(
    page_title="EuBI-Bridge GUI",
    page_icon="🔬",
    layout="wide"
)

# Inject CSS for cross-platform scrolling (Windows/Mac/Linux)
st.markdown("""
    <style>
        /* Main content area scrolling */
        [data-testid="stAppViewContainer"] {
            overflow-y: auto !important;
            height: 100vh !important;
        }
        
        /* Main content block scrolling */
        section.main > div {
            overflow-y: auto !important;
            max-height: 100vh;
        }
        
        /* Sidebar extends to full viewport height with dynamic sizing */
        [data-testid="stSidebar"] {
            height: auto !important;
            min-height: 100vh !important;
            min-height: 100dvh !important;
            display: flex !important;
            flex-direction: column !important;
        }
        
        section[data-testid="stSidebar"] > div {
            flex: 1 1 auto !important;
            max-height: 100vh !important;
            max-height: 100dvh !important;
            overflow-y: auto !important;
            display: flex !important;
            flex-direction: column !important;
        }
        
        /* Sidebar inner content scrolls within viewport */
        [data-testid="stSidebarContent"] {
            flex: 1 1 auto !important;
            min-height: 0 !important;
            overflow-y: auto !important;
        }
        
        /* Prevent horizontal overflow */
        .main .block-container {
            max-width: 100%;
            overflow-x: hidden;
        }
        
        body {
            overflow-x: hidden;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🔬 EuBI-Bridge: OME-Zarr Tool")

operation_mode = st.radio(
    "Select Operation:",
    ["🔄 Convert to OME-Zarr", "🔍 Inspect/Visualize/Edit OME-Zarr"],
    horizontal=True
)

if operation_mode == "🔄 Convert to OME-Zarr":
    st.markdown("Convert image collections to OME-Zarr format with extensive filtering and parallel processing")
elif operation_mode == "🔍 Inspect/Visualize/Edit OME-Zarr":
    st.markdown("Inspect, visualize, and edit OME-Zarr files - adjust channels, pixel metadata, and more")

if 'bridge' not in st.session_state:
    st.session_state.bridge = EuBIBridge()

bridge = st.session_state.bridge

if operation_mode == "🔍 Inspect/Visualize/Edit OME-Zarr":
    visual_channel_editor.render_sidebar_file_browser()
else:
    from eubi_bridge.views.pixel_metadata import SPACE_UNITS, TIME_UNITS
    
    if 'conv_nav_counter' not in st.session_state:
        st.session_state.conv_nav_counter = 0
    if 'conv_input_browse_path' not in st.session_state:
        st.session_state.conv_input_browse_path = os.path.expanduser("~")
    if 'conv_output_browse_path' not in st.session_state:
        st.session_state.conv_output_browse_path = os.path.expanduser("~")
    if 'conv_selected_input' not in st.session_state:
        st.session_state.conv_selected_input = ""
    if 'conv_selected_output' not in st.session_state:
        st.session_state.conv_selected_output = os.path.expanduser("~")
    if 'conv_includes' not in st.session_state:
        st.session_state.conv_includes = ""
    if 'conv_excludes' not in st.session_state:
        st.session_state.conv_excludes = ""
    if 'conv_show_input_browser' not in st.session_state:
        st.session_state.conv_show_input_browser = False
    if 'conv_show_output_browser' not in st.session_state:
        st.session_state.conv_show_output_browser = False
    
    def matches_filter(name, includes, excludes):
        if includes:
            include_patterns = [p.strip() for p in includes.split(',') if p.strip()]
            if include_patterns and not any(p in name for p in include_patterns):
                return False
        if excludes:
            exclude_patterns = [p.strip() for p in excludes.split(',') if p.strip()]
            if any(p in name for p in exclude_patterns):
                return False
        return True
    
    nav_counter = st.session_state.conv_nav_counter
    
    def on_input_path_change():
        new_val = st.session_state.conv_input_text
        if new_val:
            clean_path = os.path.expanduser(new_val.strip())
            if os.path.exists(clean_path):
                st.session_state.conv_selected_input = clean_path
                st.session_state.conv_input_browse_path = clean_path if os.path.isdir(clean_path) else os.path.dirname(clean_path)
                st.session_state.conv_input_valid = True
            else:
                st.session_state.conv_input_valid = False
        else:
            st.session_state.conv_input_valid = None
    
    if 'conv_input_text' not in st.session_state:
        st.session_state.conv_input_text = st.session_state.conv_selected_input
    if 'conv_input_valid' not in st.session_state:
        st.session_state.conv_input_valid = None
    if 'conv_input_pending' not in st.session_state:
        st.session_state.conv_input_pending = None
    
    if st.session_state.conv_input_pending is not None:
        st.session_state.conv_input_text = st.session_state.conv_input_pending
        st.session_state.conv_input_pending = None
    
    with st.sidebar.container(border=True):
        st.markdown("**📂 Input Path**")
        
        st.text_input(
            "Input:",
            key="conv_input_text",
            label_visibility="collapsed",
            placeholder="Enter path or browse...",
            on_change=on_input_path_change
        )
        
        if st.session_state.conv_input_valid is False:
            st.warning("Path not found")
        
        if st.session_state.conv_selected_input and os.path.exists(st.session_state.conv_selected_input):
            if os.path.isdir(st.session_state.conv_selected_input):
                st.success("✓ Directory selected")
            else:
                st.success("✓ File selected")
        
        if st.button("Browse", key="conv_browse_input_btn", use_container_width=True):
            st.session_state.conv_show_input_browser = not st.session_state.conv_show_input_browser
            st.session_state.conv_show_output_browser = False
        
        col_inc, col_exc = st.columns(2)
        with col_inc:
            st.session_state.conv_includes = st.text_input(
                "Include:",
                value=st.session_state.conv_includes,
                help="Patterns to include (e.g., .tif,.png)",
                key=f"conv_includes_input_{nav_counter}",
                placeholder="e.g., .tif"
            )
        with col_exc:
            st.session_state.conv_excludes = st.text_input(
                "Exclude:",
                value=st.session_state.conv_excludes,
                help="Patterns to exclude",
                key=f"conv_excludes_input_{nav_counter}",
                placeholder="e.g., _thumb"
            )
        
        if st.session_state.conv_show_input_browser:
            current_path = st.session_state.conv_input_browse_path
            
            st.caption(f"📁 {current_path}")
            
            nav_col1, nav_col2 = st.columns(2)
            with nav_col1:
                if st.button("⬆ Parent", key="conv_input_parent", use_container_width=True):
                    parent = os.path.dirname(current_path)
                    if parent and parent != current_path:
                        st.session_state.conv_input_browse_path = parent
                        st.session_state.conv_selected_input = parent
                        st.session_state.conv_input_pending = parent
                        st.session_state.conv_input_valid = True
                        st.rerun()
            with nav_col2:
                if st.button("🏠 Home", key="conv_input_home", use_container_width=True):
                    home_path = os.path.expanduser("~")
                    st.session_state.conv_input_browse_path = home_path
                    st.session_state.conv_selected_input = home_path
                    st.session_state.conv_input_pending = home_path
                    st.session_state.conv_input_valid = True
                    st.rerun()
            
            try:
                items = []
                includes = st.session_state.conv_includes
                excludes = st.session_state.conv_excludes
                filter_active = bool(includes or excludes)
                
                if os.path.exists(current_path) and os.path.isdir(current_path):
                    for item in sorted(os.listdir(current_path)):
                        if item.startswith('.'):
                            continue
                        
                        if not matches_filter(item, includes, excludes):
                            continue
                        
                        item_path = os.path.join(current_path, item)
                        is_dir = os.path.isdir(item_path)
                        items.append({'name': item, 'path': item_path, 'is_dir': is_dir})
                    
                    filter_note = " (filtered)" if filter_active else ""
                    
                    if items:
                        with st.expander(f"Contents ({len(items)}){filter_note}", expanded=True):
                            for idx, item in enumerate(items[:30]):
                                if item['is_dir']:
                                    col1, col2, col3 = st.columns([5, 2, 2])
                                    with col1:
                                        st.markdown(f"<div style='word-wrap:break-word;overflow-wrap:anywhere;'>📁 {item['name']}</div>", unsafe_allow_html=True)
                                    with col2:
                                        if st.button("Enter", key=f"conv_in_enter_{idx}"):
                                            st.session_state.conv_input_browse_path = item['path']
                                            st.session_state.conv_selected_input = item['path']
                                            st.session_state.conv_input_pending = item['path']
                                            st.session_state.conv_input_valid = True
                                            st.session_state.conv_includes = ""
                                            st.session_state.conv_excludes = ""
                                            st.rerun()
                                    with col3:
                                        if st.button("Select", key=f"conv_in_sel_dir_{idx}"):
                                            st.session_state.conv_selected_input = item['path']
                                            st.session_state.conv_input_pending = item['path']
                                            st.session_state.conv_input_valid = True
                                            st.session_state.conv_show_input_browser = False
                                            st.rerun()
                                else:
                                    col1, col2 = st.columns([7, 2])
                                    with col1:
                                        st.markdown(f"<div style='word-wrap:break-word;overflow-wrap:anywhere;'>📄 {item['name']}</div>", unsafe_allow_html=True)
                                    with col2:
                                        if st.button("Select", key=f"conv_in_sel_{idx}"):
                                            st.session_state.conv_selected_input = item['path']
                                            st.session_state.conv_input_pending = item['path']
                                            st.session_state.conv_input_valid = True
                                            st.session_state.conv_show_input_browser = False
                                            st.rerun()
                    elif filter_active:
                        st.info("No items match current filters")
            except Exception as e:
                st.error(f"Error: {e}")
    
    def on_output_path_change():
        new_val = st.session_state.conv_output_text
        if new_val:
            clean_path = os.path.expanduser(new_val.strip())
            st.session_state.conv_selected_output = clean_path
            if os.path.isdir(clean_path):
                st.session_state.conv_output_browse_path = clean_path
    
    if 'conv_output_text' not in st.session_state:
        st.session_state.conv_output_text = st.session_state.conv_selected_output
    if 'conv_output_pending' not in st.session_state:
        st.session_state.conv_output_pending = None
    
    if st.session_state.conv_output_pending is not None:
        st.session_state.conv_output_text = st.session_state.conv_output_pending
        st.session_state.conv_output_pending = None
    
    with st.sidebar.container(border=True):
        st.markdown("**📁 Output Path**")
        st.text_input(
            "Output:",
            key="conv_output_text",
            label_visibility="collapsed",
            placeholder="Enter path or browse...",
            on_change=on_output_path_change
        )
        
        if st.session_state.conv_selected_output:
            if os.path.exists(st.session_state.conv_selected_output):
                st.success("✓ Directory exists")
            else:
                st.info("Will be created")
        
        if st.button("Browse", key="conv_browse_output_btn", use_container_width=True):
            st.session_state.conv_show_output_browser = not st.session_state.conv_show_output_browser
            st.session_state.conv_show_input_browser = False
        
        if st.session_state.conv_show_output_browser:
            current_path = st.session_state.conv_output_browse_path
            
            # Auto-select the current directory being browsed
            st.session_state.conv_selected_output = current_path
            
            st.caption(f"📁 {current_path}")
            
            nav_col1, nav_col2 = st.columns(2)
            with nav_col1:
                if st.button("⬆ Parent", key="conv_output_parent", use_container_width=True):
                    parent = os.path.dirname(current_path)
                    if parent and parent != current_path:
                        st.session_state.conv_output_browse_path = parent
                        st.session_state.conv_selected_output = parent
                        st.session_state.conv_output_pending = parent
                        st.session_state.conv_nav_counter += 1
                        st.rerun()
            with nav_col2:
                if st.button("🏠 Home", key="conv_output_home", use_container_width=True):
                    home = os.path.expanduser("~")
                    st.session_state.conv_output_browse_path = home
                    st.session_state.conv_selected_output = home
                    st.session_state.conv_output_pending = home
                    st.session_state.conv_nav_counter += 1
                    st.rerun()
            
            col_new, col_create = st.columns([3, 1])
            with col_new:
                new_folder = st.text_input("New folder:", key="conv_new_folder", label_visibility="collapsed", placeholder="New folder name...")
            with col_create:
                if st.button("Create", key="conv_create_folder"):
                    if new_folder:
                        new_path = os.path.join(current_path, new_folder)
                        try:
                            os.makedirs(new_path, exist_ok=True)
                            st.session_state.conv_output_browse_path = new_path
                            st.session_state.conv_output_pending = new_path
                            st.session_state.conv_nav_counter += 1
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
            
            try:
                items = []
                if os.path.exists(current_path) and os.path.isdir(current_path):
                    for item in sorted(os.listdir(current_path)):
                        if item.startswith('.'):
                            continue
                        item_path = os.path.join(current_path, item)
                        if os.path.isdir(item_path):
                            items.append({'name': item, 'path': item_path})
                    
                    if items:
                        with st.expander(f"Folders ({len(items)})", expanded=True):
                            for idx, item in enumerate(items[:30]):
                                col1, col2, col3 = st.columns([5, 2, 2])
                                with col1:
                                    st.markdown(f"<div style='word-wrap:break-word;overflow-wrap:anywhere;'>📁 {item['name']}</div>", unsafe_allow_html=True)
                                with col2:
                                    if st.button("Enter", key=f"conv_out_enter_{idx}"):
                                        st.session_state.conv_output_browse_path = item['path']
                                        st.session_state.conv_output_pending = item['path']
                                        st.rerun()
                                with col3:
                                    if st.button("Select", key=f"conv_out_sel_{idx}"):
                                        st.session_state.conv_selected_output = item['path']
                                        st.session_state.conv_output_text = item['path']
                                        st.session_state.conv_output_pending = item['path']
                                        st.session_state.conv_show_output_browser = False
                                        st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    with st.sidebar.expander("Config Management"):
        config_action = st.radio("Action:", ["Use Current", "Load", "Save", "Reset"], key="conv_config_action", horizontal=True)
        
        if config_action == "Load":
            config_file = st.text_input("Config path:", f"{os.path.expanduser('~')}/.eubi_bridge/config.json", key="conv_load_path")
            if st.button("Load Config", key="conv_load_btn"):
                try:
                    with open(config_file, 'r') as f:
                        st.session_state.loaded_config = json.load(f)
                        st.success(f"Loaded")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        elif config_action == "Save":
            config_file = st.text_input("Config path:", "config.json", key="conv_save_path")
            if st.button("Save Config", key="conv_save_btn"):
                try:
                    config_to_save = {
                        'cluster': dict(bridge.config['cluster']),
                        'readers': dict(bridge.config['readers']),
                        'conversion': dict(bridge.config['conversion']),
                        'downscale': dict(bridge.config['downscale'])
                    }
                    with open(config_file, 'w') as f:
                        json.dump(config_to_save, f, indent=2)
                    st.success(f"Saved")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        elif config_action == "Reset":
            if st.button("Reset to Defaults", key="conv_reset_btn"):
                bridge.reset_config()
                st.success("Reset")
                st.rerun()
    
    input_path = st.session_state.conv_selected_input
    output_path = st.session_state.conv_selected_output
    includes = st.session_state.conv_includes
    excludes = st.session_state.conv_excludes

if operation_mode == "🔄 Convert to OME-Zarr":
    tab1, tab2 = st.tabs(["⚙️ Conversion Parameters", "▶️ Run Conversion"])
    
    with tab1:
        st.header("Conversion Parameters")
        
        with st.expander("Concatenation Settings (modify only for aggregative conversions)", expanded=False):
            st.info("Use these settings only when combining multiple source files into a single output. For simple single-file conversions, leave this section collapsed.")
            
            col_tags, col_axes = st.columns([2, 1])
        
            with col_tags:
                st.markdown("**Dimension Tags**")
                tag_col1, tag_col2 = st.columns(2)
                with tag_col1:
                    time_tag = st.text_input("Time Tag", help="Pattern to identify time series (e.g., 'T', 't')", key="conv_time_tag")
                    channel_tag = st.text_input("Channel Tag", help="Pattern for channels (e.g., 'C', 'channel')", key="conv_channel_tag")
                    x_tag = st.text_input("X Tag", help="Pattern for X dimension", key="conv_x_tag")
                with tag_col2:
                    z_tag = st.text_input("Z Tag", help="Pattern for Z-stack dimension", key="conv_z_tag")
                    y_tag = st.text_input("Y Tag", help="Pattern for Y dimension", key="conv_y_tag")
        
            with col_axes:
                st.markdown("**Axes to Concatenate**")
                concatenation_axes = st.text_input(
                    "Concatenation Axes",
                    help="Axes to concatenate along (e.g., 'tc' for time and channel). Leave empty for no concatenation.",
                    key="conv_concat_axes"
                )
        
        param_tab1, param_tab2, param_tab3, param_tab4, param_tab5 = st.tabs([
            "🔧 Cluster",
            "📖 Reader",
            "🔄 Conversion",
            "📊 Downscaling",
            "📐 Metadata"
        ])
    
        with param_tab1:
            with st.container(height=450, border=True):
                st.subheader("Parallel Processing Configuration")
    
                col1, col2 = st.columns(2)
    
                with col1:
                    max_workers = st.number_input(
                        "Max Workers",
                        min_value=1,
                        max_value=64,
                        value=bridge.config['cluster']['max_workers'],
                        help="Number of parallel workers"
                    )
    
                    queue_size = st.number_input(
                        "Queue Size",
                        min_value=1,
                        max_value=32,
                        value=bridge.config['cluster']['queue_size'],
                        help="Number of batches to process in parallel"
                    )
    
                    max_concurrency = st.number_input(
                        "Max Concurrency",
                        min_value=1,
                        max_value=64,
                        value=bridge.config['cluster']['max_concurrency'],
                        help="Maximum concurrent operations"
                    )
    
                with col2:
                    region_size_mb = st.text_input(
                        "Region Size (MB)",
                        value=bridge.config['cluster']['region_size_mb'],
                        help="Memory limit per batch (e.g., '1GB', '512MB')"
                    )
    
                    memory_per_worker = st.text_input(
                        "Memory per Worker",
                        value=bridge.config['cluster']['memory_per_worker'],
                        help="Memory allocation per worker (e.g., '10GB')"
                    )
    
                    on_local_cluster = st.checkbox(
                        "Use Local Dask Cluster",
                        value=bridge.config['cluster']['on_local_cluster'],
                        help="Enable Dask distributed cluster for large-scale processing"
                    )
    
                    on_slurm = st.checkbox(
                        "Use SLURM Cluster",
                        value=bridge.config['cluster']['on_slurm'],
                        help="Enable SLURM cluster for HPC environments"
                    )
    
        with param_tab2:
            with st.container(height=450, border=True):
                st.subheader("Reader Configuration")
    
                col1, col2, col3 = st.columns(3)
    
                def validate_index_input(value, field_name):
                    """Validate index input: integer or comma-separated integers"""
                    value = value.strip()
                    if not value:
                        return False, f"{field_name} cannot be empty"
                    parts = [p.strip() for p in value.split(',')]
                    for part in parts:
                        if not part.lstrip('-').isdigit():
                            return False, f"{field_name} must be an integer or comma-separated integers (e.g., '0' or '0,1,2')"
                    return True, None
    
                with col1:
                    read_all_scenes = st.checkbox(
                        "Read all scenes",
                        value=bridge.config['readers']['scene_index'] == 'all',
                        help="Process all scenes in the file"
                    )
    
                    default_scene = bridge.config['readers']['scene_index']
                    if default_scene == 'all':
                        default_scene = 0
    
                    scene_index = st.text_input(
                        "Scene Index",
                        value=str(default_scene),
                        help="Integer or comma-separated integers (e.g., '0' or '0,1,2')",
                        disabled=read_all_scenes
                    )
                    if not read_all_scenes:
                        scene_valid, scene_error = validate_index_input(scene_index, "Scene Index")
                        if not scene_valid:
                            st.error(scene_error)
    
                    read_all_tiles = st.checkbox(
                        "Read all tiles",
                        value=bridge.config['readers']['mosaic_tile_index'] == 'all',
                        help="Process all mosaic tiles in the file"
                    )
    
                    default_mosaic = bridge.config['readers']['mosaic_tile_index']
                    if default_mosaic == 'all':
                        default_mosaic = 0
    
                    mosaic_tile_index = st.text_input(
                        "Mosaic Tile Index",
                        value=str(default_mosaic),
                        help="Integer or comma-separated integers (e.g., '0' or '0,1,2')",
                        disabled=read_all_tiles
                    )
                    if not read_all_tiles:
                        mosaic_valid, mosaic_error = validate_index_input(mosaic_tile_index, "Mosaic Tile Index")
                        if not mosaic_valid:
                            st.error(mosaic_error)
    
                    as_mosaic = st.checkbox(
                        "Read as Mosaic",
                        value=bridge.config['readers']['as_mosaic']
                    )
    
                with col2:
                    view_index = st.number_input(
                        "View Index",
                        min_value=0,
                        value=bridge.config['readers']['view_index']
                    )
    
                    phase_index = st.number_input(
                        "Phase Index",
                        min_value=0,
                        value=bridge.config['readers']['phase_index']
                    )
    
                    illumination_index = st.number_input(
                        "Illumination Index",
                        min_value=0,
                        value=bridge.config['readers']['illumination_index']
                    )
    
                with col3:
                    rotation_index = st.number_input(
                        "Rotation Index",
                        min_value=0,
                        value=bridge.config['readers']['rotation_index']
                    )
    
                    sample_index = st.number_input(
                        "Sample Index",
                        min_value=0,
                        value=bridge.config['readers']['sample_index']
                    )
    
        with param_tab3:
            with st.container(height=450, border=True):
                st.subheader("Conversion Settings")
    
                col1, col2 = st.columns(2)
    
                with col1:
                    st.markdown("**General Settings**")
    
                    zarr_format = st.selectbox(
                        "Zarr Format",
                        options=[2, 3],
                        index=0 if bridge.config['conversion']['zarr_format'] == 2 else 1,
                        help="Zarr format version"
                    )
    
                    dtype = st.selectbox(
                        "Data Type",
                        options=['auto', 'uint8', 'uint16', 'uint32', 'float32', 'float64'],
                        index=0,
                        help="Output data type"
                    )
    
                    # Compression configuration with dynamic parameters
                    compressor, compressor_params = render_compression_config(key_prefix="conv_")
    
                    verbose = st.checkbox(
                        "Verbose Output",
                        value=bridge.config['conversion']['verbose'],
                        help="Enable detailed logging"
                    )
    
                    overwrite = st.checkbox(
                        "Overwrite Existing",
                        value=bridge.config['conversion']['overwrite'],
                        help="Overwrite existing output files"
                    )
    
                    squeeze = st.checkbox(
                        "Squeeze Dimensions",
                        value=bridge.config['conversion']['squeeze'],
                        help="Remove singleton dimensions"
                    )
    
                    save_omexml = st.checkbox(
                        "Save OME-XML",
                        value=bridge.config['conversion']['save_omexml'],
                        help="Save OME-XML metadata"
                    )
    
                    override_channel_names = st.checkbox(
                        "Override Channel Names",
                        value=bridge.config['conversion']['override_channel_names'],
                        help="Use custom channel names from tags"
                    )
    
                    skip_dask = st.checkbox(
                        "Skip Dask",
                        value=bridge.config['conversion'].get('skip_dask', False),
                        help="Perform one-to-one TIFF file conversions without using dask arrays"
                    )
    
                with col2:
                    st.markdown("**Chunking Settings**")
    
                    auto_chunk = st.checkbox(
                        "Auto Chunk",
                        value=bridge.config['conversion']['auto_chunk'],
                        help="Automatically calculate chunk dimensions based on target chunk size"
                    )
    
                    target_chunk_mb = st.number_input(
                        "Target Chunk Size (MB)",
                        min_value=0.1,
                        max_value=100.0,
                        value=float(bridge.config['conversion']['target_chunk_mb']),
                        step=0.1,
                        help="Target size for each chunk (used for auto-chunking)",
                        disabled=not auto_chunk
                    )
    
                    st.markdown("**Manual Chunk Dimensions**")
                    if auto_chunk:
                        st.caption("Disabled - dimensions calculated automatically from target size")
    
                    col_t, col_c = st.columns(2)
                    with col_t:
                        time_chunk = st.number_input("Time Chunk", min_value=1, value=bridge.config['conversion']['time_chunk'],
                                                     disabled=auto_chunk)
                        z_chunk = st.number_input("Z Chunk", min_value=1, value=bridge.config['conversion']['z_chunk'],
                                                  disabled=auto_chunk)
                        x_chunk = st.number_input("X Chunk", min_value=1, value=bridge.config['conversion']['x_chunk'],
                                                  disabled=auto_chunk)
    
                    with col_c:
                        channel_chunk = st.number_input("Channel Chunk", min_value=1,
                                                        value=bridge.config['conversion']['channel_chunk'], disabled=auto_chunk)
                        y_chunk = st.number_input("Y Chunk", min_value=1, value=bridge.config['conversion']['y_chunk'],
                                                  disabled=auto_chunk)
    
                    st.markdown("**Sharding Coefficients**")
                    if zarr_format == 2:
                        st.caption("Disabled - sharding is only available for Zarr format 3")
    
                    col_s1, col_s2 = st.columns(2)
                    with col_s1:
                        time_shard_coef = st.number_input("Time Shard Coef", min_value=1,
                                                          value=bridge.config['conversion']['time_shard_coef'],
                                                          disabled=(zarr_format == 2))
                        z_shard_coef = st.number_input("Z Shard Coef", min_value=1,
                                                       value=bridge.config['conversion']['z_shard_coef'],
                                                       disabled=(zarr_format == 2))
                        x_shard_coef = st.number_input("X Shard Coef", min_value=1,
                                                       value=bridge.config['conversion']['x_shard_coef'],
                                                       disabled=(zarr_format == 2))
    
                    with col_s2:
                        channel_shard_coef = st.number_input("Channel Shard Coef", min_value=1,
                                                             value=bridge.config['conversion']['channel_shard_coef'],
                                                             disabled=(zarr_format == 2))
                        y_shard_coef = st.number_input("Y Shard Coef", min_value=1,
                                                       value=bridge.config['conversion']['y_shard_coef'],
                                                       disabled=(zarr_format == 2))
    
                    st.markdown("**Dimension Ranges (optional cropping)**")
                    st.caption("Format: start,end (e.g., '0,100'). Leave empty for full range.")
    
                    time_range = st.text_input("Time Range", help="Time dimension range")
                    channel_range = st.text_input("Channel Range", help="Channel dimension range")
                    z_range = st.text_input("Z Range", help="Z dimension range")
                    y_range = st.text_input("Y Range", help="Y dimension range")
                    x_range = st.text_input("X Range", help="X dimension range")
    
        with param_tab4:
            with st.container(height=450, border=True):
                st.subheader("Downscaling/Pyramid Settings")
    
                n_layers_auto = st.checkbox(
                    "Auto-detect Number of Layers",
                    value=bridge.config['downscale']['n_layers'] is None,
                    help="Let the tool automatically determine the optimal number of pyramid levels based on minimum dimension size and scale factors"
                )
    
                col1, col2 = st.columns(2)
    
                with col1:
                    st.markdown("**Manual Layer Count**")
                    n_layers = st.number_input(
                        "Number of Layers",
                        min_value=1,
                        max_value=10,
                        value=bridge.config['downscale']['n_layers'] if bridge.config['downscale']['n_layers'] else 3,
                        help="Number of pyramid levels",
                        disabled=n_layers_auto
                    )
                    if n_layers_auto:
                        n_layers = None
    
                    st.markdown("**Auto-detection Parameters**")
                    min_dimension_size = st.number_input(
                        "Minimum Dimension Size",
                        min_value=1,
                        max_value=512,
                        value=bridge.config['downscale']['min_dimension_size'],
                        help="Stop downscaling when reaching this size (used for auto-detection)",
                        disabled=not n_layers_auto
                    )
    
                with col2:
                    st.markdown("**Scale Factors per Dimension**")
                    st.caption("Used for auto-detection when enabled" if n_layers_auto else "Applied to each pyramid level")
    
                    time_scale_factor = st.number_input("Time Scale Factor", min_value=1,
                                                        value=bridge.config['downscale']['time_scale_factor'])
                    channel_scale_factor = st.number_input("Channel Scale Factor", min_value=1,
                                                           value=bridge.config['downscale']['channel_scale_factor'])
                    z_scale_factor = st.number_input("Z Scale Factor", min_value=1,
                                                     value=bridge.config['downscale']['z_scale_factor'])
                    y_scale_factor = st.number_input("Y Scale Factor", min_value=1,
                                                     value=bridge.config['downscale']['y_scale_factor'])
                    x_scale_factor = st.number_input("X Scale Factor", min_value=1,
                                                     value=bridge.config['downscale']['x_scale_factor'])
        
        with param_tab5:
            with st.container(height=450, border=True):
                st.subheader("Metadata Settings")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    metadata_reader = st.selectbox(
                        "Metadata Reader",
                        options=['bfio', 'bioio'],
                        index=0,
                        help="Library used to read metadata from input files",
                        key="conv_metadata_reader"
                    )
                
                with col2:
                    channel_intensity_limits = st.selectbox(
                        "Channel Intensity Limits",
                        options=['From Datatype', 'From Array Values'],
                        index=0,
                        help="How to determine channel intensity display ranges",
                        key="conv_intensity_limits"
                    )
                
                unit_default = "(use original)"
                time_unit_options = [unit_default] + TIME_UNITS[1:]
                space_unit_options = [unit_default] + SPACE_UNITS[1:]
                
                with st.container(border=True):
                    st.markdown("**Metadata Override**")
                    st.caption("Override physical scale and units from input files. Leave empty to keep original values.")
                    
                    col_dim, col_scale, col_unit = st.columns([1, 2, 2])
                    with col_dim:
                        st.markdown("**Dim**")
                    with col_scale:
                        st.markdown("**Scale**")
                    with col_unit:
                        st.markdown("**Unit**")
                    
                    col_dim, col_scale, col_unit = st.columns([1, 2, 2])
                    with col_dim:
                        st.markdown("T")
                    with col_scale:
                        time_scale = st.text_input("Time Scale", label_visibility="collapsed", placeholder="e.g., 1.0", key="conv_time_scale")
                    with col_unit:
                        time_unit = st.selectbox("Time Unit", options=time_unit_options, index=0, label_visibility="collapsed", key="conv_time_unit")
                    
                    col_dim, col_scale, col_unit = st.columns([1, 2, 2])
                    with col_dim:
                        st.markdown("Z")
                    with col_scale:
                        z_scale = st.text_input("Z Scale", label_visibility="collapsed", placeholder="e.g., 1.0", key="conv_z_scale")
                    with col_unit:
                        z_unit = st.selectbox("Z Unit", options=space_unit_options, index=0, label_visibility="collapsed", key="conv_z_unit")
                    
                    col_dim, col_scale, col_unit = st.columns([1, 2, 2])
                    with col_dim:
                        st.markdown("Y")
                    with col_scale:
                        y_scale = st.text_input("Y Scale", label_visibility="collapsed", placeholder="e.g., 0.5", key="conv_y_scale")
                    with col_unit:
                        y_unit = st.selectbox("Y Unit", options=space_unit_options, index=0, label_visibility="collapsed", key="conv_y_unit")
                    
                    col_dim, col_scale, col_unit = st.columns([1, 2, 2])
                    with col_dim:
                        st.markdown("X")
                    with col_scale:
                        x_scale = st.text_input("X Scale", label_visibility="collapsed", placeholder="e.g., 0.5", key="conv_x_scale")
                    with col_unit:
                        x_unit = st.selectbox("X Unit", options=space_unit_options, index=0, label_visibility="collapsed", key="conv_x_unit")

    with tab2:
        st.header("Run Conversion")
    
        st.info("Review your settings and start the conversion process")
    
        col1, col2 = st.columns([2, 1])
    
        with col1:
            st.subheader("Summary")
            st.write(f"**Input:** {input_path if input_path else 'Not set'}")
            st.write(f"**Output:** {output_path if output_path else 'Not set'}")
            st.write(f"**Includes:** {includes if includes else 'All files'}")
            st.write(f"**Excludes:** {excludes if excludes else 'None'}")
            st.write(f"**Concatenation:** {concatenation_axes if concatenation_axes else 'None'}")
            st.write(f"**Max Workers:** {max_workers}")
            st.write(f"**Zarr Format:** {zarr_format}")
    
        with col2:
            st.subheader("Actions")
    
            if st.button("🚀 Start Conversion", type="primary", use_container_width=True):
                if not input_path:
                    st.error("Please provide an input path")
                elif not output_path and not input_path.endswith(('.csv', '.xlsx')):
                    st.error("Please provide an output path")
                else:
                    def parse_range(range_str):
                        if not range_str:
                            return None
                        parts = range_str.split(',')
                        if len(parts) == 2:
                            return (int(parts[0]), int(parts[1]))
                        return None
    
    
                    def parse_index_value(value):
                        """Parse index value: returns int or list of ints"""
                        value = value.strip()
                        parts = [p.strip() for p in value.split(',')]
                        if len(parts) == 1:
                            return int(parts[0])
                        return [int(p) for p in parts]
    
    
                    scene_idx = 'all' if read_all_scenes else parse_index_value(scene_index)
                    mosaic_idx = 'all' if read_all_tiles else parse_index_value(mosaic_tile_index)
    
                    kwargs = {
                        'max_workers': max_workers,
                        'queue_size': queue_size,
                        'region_size_mb': region_size_mb,
                        'max_concurrency': max_concurrency,
                        'memory_per_worker': memory_per_worker,
                        'on_local_cluster': on_local_cluster,
                        'on_slurm': on_slurm,
                        'scene_index': scene_idx,
                        'mosaic_tile_index': mosaic_idx,
                        'as_mosaic': as_mosaic,
                        'view_index': view_index,
                        'phase_index': phase_index,
                        'illumination_index': illumination_index,
                        'rotation_index': rotation_index,
                        'sample_index': sample_index,
                        'verbose': verbose,
                        'zarr_format': zarr_format,
                        'auto_chunk': auto_chunk,
                        'time_chunk': time_chunk,
                        'channel_chunk': channel_chunk,
                        'z_chunk': z_chunk,
                        'y_chunk': y_chunk,
                        'x_chunk': x_chunk,
                        'time_shard_coef': time_shard_coef,
                        'channel_shard_coef': channel_shard_coef,
                        'z_shard_coef': z_shard_coef,
                        'y_shard_coef': y_shard_coef,
                        'x_shard_coef': x_shard_coef,
                        'time_range': parse_range(time_range),
                        'channel_range': parse_range(channel_range),
                        'z_range': parse_range(z_range),
                        'y_range': parse_range(y_range),
                        'x_range': parse_range(x_range),
                        'compressor': compressor,
                        'overwrite': overwrite,
                        'override_channel_names': override_channel_names,
                        'channel_intensity_limits': 'from_dtype' if channel_intensity_limits == 'From Datatype' else 'from_array',
                        'metadata_reader': metadata_reader,
                        'save_omexml': save_omexml,
                        'squeeze': squeeze,
                        'skip_dask': skip_dask,
                        'dtype': dtype if dtype != 'auto' else 'auto',
                        'n_layers': n_layers,
                        'min_dimension_size': min_dimension_size,
                        'time_scale_factor': time_scale_factor,
                        'channel_scale_factor': channel_scale_factor,
                        'z_scale_factor': z_scale_factor,
                        'y_scale_factor': y_scale_factor,
                        'x_scale_factor': x_scale_factor,
                        'compressor': compressor,
                        'compressor_params': compressor_params,
                    }

                    import pprint
                    #pprint.pprint(kwargs)
    
                    if auto_chunk:
                        kwargs['target_chunk_mb'] = target_chunk_mb
                    
                    if time_scale and time_scale.strip():
                        try:
                            kwargs['time_scale'] = float(time_scale)
                        except ValueError:
                            pass
                    if z_scale and z_scale.strip():
                        try:
                            kwargs['z_scale'] = float(z_scale)
                        except ValueError:
                            pass
                    if y_scale and y_scale.strip():
                        try:
                            kwargs['y_scale'] = float(y_scale)
                        except ValueError:
                            pass
                    if x_scale and x_scale.strip():
                        try:
                            kwargs['x_scale'] = float(x_scale)
                        except ValueError:
                            pass
                    
                    unit_default = "(use original)"
                    if time_unit and time_unit != unit_default:
                        kwargs['time_unit'] = time_unit
                    if z_unit and z_unit != unit_default:
                        kwargs['z_unit'] = z_unit
                    if y_unit and y_unit != unit_default:
                        kwargs['y_unit'] = y_unit
                    if x_unit and x_unit != unit_default:
                        kwargs['x_unit'] = x_unit
    
                    st.markdown("### Conversion Log")
                    status_container = st.empty()
                    log_container = st.container()
                    status_container.info("🔄 Conversion in progress...")
    
                    mp_manager = multiprocessing.Manager()
                    log_queue = mp_manager.Queue()
    
                    handler = QueueHandler(log_queue)
                    handler.setLevel(logging.INFO)
    
                    root_logger = logging.getLogger()
                    root_logger.addHandler(handler)
                    root_logger.setLevel(logging.INFO)
    
                    eubi_logger = logging.getLogger("eubi_bridge")
                    # eubi_logger.addHandler(handler)
                    eubi_logger.setLevel(logging.INFO)
                    eubi_logger.propagate = True
    
                    from concurrent.futures import ProcessPoolExecutor as OriginalPPE
                    from concurrent.futures import ThreadPoolExecutor as OriginalTPE
    
                    class LoggingProcessPoolExecutor(OriginalPPE):
                        """ProcessPoolExecutor that sets up logging in child processes"""
    
                        def __init__(self, max_workers=None, **kw):
                            kw['initializer'] = setup_mp_logging
                            kw['initargs'] = (log_queue,)
                            super().__init__(max_workers, **kw)
    
                    class LoggingThreadPoolExecutor(OriginalTPE):
                        """ThreadPoolExecutor that sets up logging in child processes"""
    
                        def __init__(self, max_workers=None, **kw):
                            kw['initializer'] = setup_mp_logging
                            kw['initargs'] = (log_queue,)
                            super().__init__(max_workers, **kw)
    
    
                    import eubi_bridge.conversion.converter as converter_module
    
                    original_ppe = converter_module.ProcessPoolExecutor
                    original_tpe = converter_module.ThreadPoolExecutor
                    converter_module.ProcessPoolExecutor = LoggingProcessPoolExecutor
                    converter_module.ThreadPoolExecutor = LoggingThreadPoolExecutor
    
                    conversion_result = {'success': False, 'error': None, 'traceback': None}
    
                    class QueueWriter:
                        """Redirect stdout/stderr to the queue for RichHandler output"""
                        def __init__(self, queue, original, is_stderr=False):
                            self.queue = queue
                            self.original = original
                            self.is_stderr = is_stderr
    
                        def write(self, text):
                            if text and text.strip():
                                self.queue.put(text.rstrip())
                            if self.original:
                                self.original.write(text)
    
                        def flush(self):
                            if self.original:
                                self.original.flush()
    
                        def isatty(self):
                            return True
    
                    def run_conversion():
                        original_stdout = sys.stdout
                        original_stderr = sys.stderr
                        sys.stdout = QueueWriter(log_queue, sys.__stdout__)
                        sys.stderr = QueueWriter(log_queue, sys.__stderr__, is_stderr=True)
                        try:
                            conversion_bridge = EuBIBridge()
                            conversion_bridge.to_zarr(
                                input_path=input_path,
                                output_path=output_path if output_path else None,
                                includes=includes.split(',') if includes else None,
                                excludes=excludes.split(',') if excludes else None,
                                time_tag=time_tag if time_tag else None,
                                channel_tag=channel_tag if channel_tag else None,
                                z_tag=z_tag if z_tag else None,
                                y_tag=y_tag if y_tag else None,
                                x_tag=x_tag if x_tag else None,
                                concatenation_axes=concatenation_axes if concatenation_axes else None,
                                **kwargs
                            )
                            pprint.pprint("Conversion kwargs:")
                            pprint.pprint(kwargs)
                            conversion_result['success'] = True
                        except Exception as e:
                            import traceback
                            conversion_result['error'] = str(e)
                            conversion_result['traceback'] = traceback.format_exc()
                        finally:
                            sys.stdout = original_stdout
                            sys.stderr = original_stderr
    
    
                    conversion_thread = threading.Thread(target=run_conversion)
                    conversion_thread.start()
    
                    from rich.console import Console
                    from rich.ansi import AnsiDecoder
                    from io import StringIO
    
    
                    def ansi_to_html(ansi_text):
                        """Convert ANSI-styled text to HTML using Rich"""
                        console = Console(file=StringIO(), force_terminal=True, record=True, width=200)
                        decoder = AnsiDecoder()
                        for line in decoder.decode(ansi_text):
                            console.print(line)
                        return console.export_html(inline_styles=True,
                                                   code_format='<pre style="font-family:Menlo,\'DejaVu Sans Mono\',consolas,\'Courier New\',monospace;white-space:pre-wrap;word-wrap:break-word">{code}</pre>')
    
    
                    def render_log_html(logs, max_lines=500):
                        """Render logs as scrollable HTML container with Rich styling"""
                        recent_logs = logs[-max_lines:] if len(logs) > max_lines else logs
                        log_text = '\n'.join(recent_logs)
    
                        try:
                            rich_html = ansi_to_html(log_text)
                        except:
                            escaped_logs = [html_module.escape(line) for line in recent_logs]
                            rich_html = '<pre>' + '\n'.join(escaped_logs) + '</pre>'
    
                        return f'''
                        <!DOCTYPE html>
                        <html>
                        <head>
                        <style>
                        body {{
                            margin: 0;
                            padding: 0;
                            background-color: #f5f3ef;
                        }}
                        .log-container {{
                            height: 380px;
                            overflow-y: auto;
                            overflow-x: hidden;
                            background-color: #f5f3ef;
                            padding: 10px;
                            font-family: 'Source Code Pro', 'Courier New', monospace;
                            font-size: 13px;
                            line-height: 1.5;
                            color: #1a1a1a;
                        }}
                        .log-container pre {{
                            margin: 0;
                            white-space: pre-wrap;
                            word-wrap: break-word;
                            word-break: break-word;
                            background: transparent !important;
                        }}
                        </style>
                        </head>
                        <body>
                        <div class="log-container" id="logDiv">{rich_html}</div>
                        <script>
                        var logDiv = document.getElementById('logDiv');
                        logDiv.scrollTop = logDiv.scrollHeight;
                        </script>
                        </body>
                        </html>
                        '''
    
    
                    log_placeholder = log_container.empty()
    
                    all_logs = []
                    while conversion_thread.is_alive():
                        try:
                            while True:
                                msg = log_queue.get_nowait()
                                all_logs.append(msg)
                        except:
                            pass
    
                        if all_logs:
                            log_placeholder.empty()
                            with log_placeholder.container():
                                components.html(render_log_html(all_logs), height=400, scrolling=False)
    
                        time.sleep(0.1)
    
                    try:
                        while True:
                            msg = log_queue.get_nowait()
                            all_logs.append(msg)
                    except:
                        pass
    
                    if all_logs:
                        log_placeholder.empty()
                        with log_placeholder.container():
                            components.html(render_log_html(all_logs), height=400, scrolling=False)
    
                    root_logger.removeHandler(handler)
                    # eubi_logger.removeHandler(handler)
    
                    try:
                        converter_module.ProcessPoolExecutor = original_ppe
                        converter_module.ThreadPoolExecutor = original_tpe
                    except:
                        pass
    
                    if conversion_result['success']:
                        status_container.success("✅ Conversion completed successfully!")
                    else:
                        status_container.error(f"❌ Conversion failed: {conversion_result['error']}")
                        if conversion_result['traceback']:
                            with st.expander("Show error details"):
                                st.code(conversion_result['traceback'])
    
            if st.button("📋 Show Full Config", use_container_width=True):
                st.json(bridge.config)
    

elif operation_mode == "🔍 Inspect/Visualize/Edit OME-Zarr":
    if 'editor_view_selection' not in st.session_state:
        st.session_state.editor_view_selection = "🎨 Channel Controls"
    
    def on_view_change():
        st.session_state.editor_view_selection = st.session_state.editor_view_radio
    
    view_selection = st.radio(
        "View:",
        ["🎨 Channel Controls", "📐 Metadata"],
        horizontal=True,
        key="editor_view_radio",
        index=0 if st.session_state.editor_view_selection == "🎨 Channel Controls" else 1,
        on_change=on_view_change,
        label_visibility="collapsed"
    )
    
    if st.session_state.editor_view_selection == "🎨 Channel Controls":
        visual_channel_editor.render(bridge)
    else:
        pixel_metadata.render(bridge)

st.markdown("---")
st.markdown("### About EuBI-Bridge")
st.caption(
    "EuBI-Bridge is a conversion tool for bioimage datasets, enabling both unary and aggregative conversion of image data collections to OME-Zarr format with extensive parallel processing capabilities.")


def main():
    """Entry point for eubi-gui command.
    
    This function is called when users run `eubi-gui` from the command line via setup.py entry_points.
    Do NOT call this when running with `streamlit run` directly.
    """
    import subprocess
    import sys
    import os
    
    # Get the path to this script
    script_path = os.path.abspath(__file__)
    
    # Launch streamlit with this script
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", script_path,
        "--logger.level=info"
    ])


# NOTE: Do NOT uncomment the line below when running with Streamlit directly.
# Streamlit handles execution automatically. Uncommenting this causes multiple browser windows.
# if __name__ == "__main__":
#     main()
