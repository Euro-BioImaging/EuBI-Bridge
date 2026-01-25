"""Shared utilities for EuBI-Bridge views"""
import streamlit as st
import streamlit.components.v1 as components
import os
import sys
import logging
import time
import threading
import multiprocessing
import html as html_module

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


def ansi_to_html(ansi_text):
    """Convert ANSI-styled text to HTML using Rich"""
    from rich.console import Console
    from rich.ansi import AnsiDecoder
    from io import StringIO
    
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


def run_operation_with_logging(operation_func, status_container, log_container):
    """Run an operation with logging support and display progress"""
    from eubi_bridge.ebridge import EuBIBridge
    
    mp_manager = multiprocessing.Manager()
    log_queue = mp_manager.Queue()

    handler = QueueHandler(log_queue)
    handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    eubi_logger = logging.getLogger("eubi_bridge")
    eubi_logger.setLevel(logging.INFO)
    eubi_logger.propagate = True

    result = {'success': False, 'error': None, 'traceback': None}

    def run_with_redirect():
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = QueueWriter(log_queue, sys.__stdout__)
        sys.stderr = QueueWriter(log_queue, sys.__stderr__, is_stderr=True)
        try:
            operation_func()
            result['success'] = True
        except Exception as e:
            import traceback
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    operation_thread = threading.Thread(target=run_with_redirect)
    operation_thread.start()

    log_placeholder = log_container.empty()
    all_logs = []

    while operation_thread.is_alive():
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

    return result


def render_path_input(key_prefix, label="Path", help_text="Path to OME-Zarr file or directory"):
    """Render a path input with optional filesystem browser"""
    browse_key = f'{key_prefix}_browse_mode'
    path_key = f'{key_prefix}_current_path'
    selected_key = f'{key_prefix}_selected_path'
    input_key = f'{key_prefix}_input'
    pending_key = f'{key_prefix}_pending_selection'
    
    if browse_key not in st.session_state:
        st.session_state[browse_key] = False
    if path_key not in st.session_state:
        st.session_state[path_key] = os.path.expanduser("~")
    if selected_key not in st.session_state:
        st.session_state[selected_key] = ""
    if pending_key not in st.session_state:
        st.session_state[pending_key] = None

    if st.session_state[pending_key] is not None:
        st.session_state[input_key] = st.session_state[pending_key]
        st.session_state[selected_key] = st.session_state[pending_key]
        st.session_state[pending_key] = None

    input_path = st.text_input(
        label,
        help=help_text,
        key=input_key
    )

    if input_path:
        st.session_state[selected_key] = input_path

    if input_path and os.path.exists(input_path):
        if os.path.isdir(input_path):
            st.success(f"Directory found")
        elif os.path.isfile(input_path) or input_path.endswith('.zarr'):
            st.success(f"Path found")
    elif input_path:
        st.warning("Path not found")

    if st.button("Browse Filesystem", key=f'{key_prefix}_browse_btn'):
        st.session_state[browse_key] = not st.session_state[browse_key]

    if st.session_state[browse_key]:
        st.markdown("""
        <style>
        [data-testid="stSidebar"] .element-container p,
        [data-testid="stSidebar"] .element-container div {
            white-space: normal !important;
            word-wrap: break-word !important;
            overflow-wrap: break-word !important;
            word-break: break-word !important;
        }
        .file-item-name {
            white-space: normal !important;
            word-wrap: break-word !important;
            overflow-wrap: break-word !important;
            word-break: break-word !important;
            display: block !important;
            max-width: 100% !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**Filesystem Browser**")

        current_path = st.session_state[path_key]

        col_nav1, col_nav2, col_nav3 = st.columns([4, 1, 1])
        with col_nav1:
            st.text(f"{current_path}")
        with col_nav2:
            if st.button("Parent", key=f'{key_prefix}_parent_btn'):
                parent_path = os.path.dirname(current_path)
                if parent_path and parent_path != current_path:
                    st.session_state[path_key] = parent_path
                    st.rerun()
        with col_nav3:
            if st.button("Home", key=f'{key_prefix}_home_btn'):
                st.session_state[path_key] = os.path.expanduser("~")
                st.rerun()

        show_hidden = st.checkbox("Show hidden files", key=f'{key_prefix}_show_hidden')

        def is_zarr_store(folder_path):
            """Check if a folder is a zarr store (contains zarr.json or .zattrs or ends with .zarr)"""
            try:
                if folder_path.endswith('.zarr'):
                    return True
                return (os.path.exists(os.path.join(folder_path, 'zarr.json')) or 
                        os.path.exists(os.path.join(folder_path, '.zattrs')))
            except:
                return False

        try:
            if not os.path.exists(current_path):
                st.error(f"Path does not exist: {current_path}")
                if st.button("Reset to Home", key=f'{key_prefix}_reset_home'):
                    st.session_state[path_key] = os.path.expanduser("~")
                    st.rerun()
            elif not os.path.isdir(current_path):
                st.error(f"Path is not a directory: {current_path}")
            else:
                items = []
                try:
                    all_items = os.listdir(current_path)
                except Exception as list_err:
                    st.error(f"Cannot list directory: {list_err}")
                    all_items = []
                
                if show_hidden:
                    visible_items = sorted(all_items)
                    hidden_count = 0
                else:
                    visible_items = sorted([f for f in all_items if not f.startswith('.')])
                    hidden_count = len(all_items) - len(visible_items)
                error_count = 0
                
                for item in visible_items:
                    item_path = os.path.join(current_path, item)
                    try:
                        is_dir = os.path.isdir(item_path)
                        is_zarr = is_zarr_store(item_path) if is_dir else False
                        items.append({'name': item, 'path': item_path, 'is_dir': is_dir, 'is_zarr': is_zarr})
                    except PermissionError:
                        error_count += 1
                        continue
                    except Exception as item_err:
                        error_count += 1
                        pass

                st.caption(f"{len(items)} items" +
                          (f" ({hidden_count} hidden)" if hidden_count > 0 else "") +
                          (f", {error_count} errors" if error_count > 0 else ""))
                
                if items:
                    st.markdown(f"**Contents:**")
                    for idx, item in enumerate(items):
                        if item['is_zarr']:
                            icon = "🗃️"
                        elif item['is_dir']:
                            icon = "📁"
                        else:
                            icon = "📄"
                        escaped_name = item['name'].replace('<', '&lt;').replace('>', '&gt;')
                        st.markdown(
                            f'<div class="file-item-name" style="font-size:0.85em;margin-bottom:2px;">'
                            f'{icon} {escaped_name}</div>',
                            unsafe_allow_html=True
                        )
                        if item['is_dir'] or item['is_zarr']:
                            btn_cols = st.columns(2)
                            with btn_cols[0]:
                                if item['is_dir']:
                                    if st.button("Enter", key=f"{key_prefix}_open_{idx}", use_container_width=True):
                                        st.session_state[path_key] = item['path']
                                        st.rerun()
                            with btn_cols[1]:
                                if item['is_zarr']:
                                    if st.button("Select", key=f"{key_prefix}_sel_{idx}", type="primary", use_container_width=True):
                                        st.session_state[pending_key] = item['path']
                                        st.session_state[browse_key] = False
                                        st.rerun()
                elif hidden_count > 0:
                    st.info(f"All {hidden_count} items are hidden (start with '.'). No visible items to show.")
                elif len(all_items) == 0:
                    st.info("Directory is empty")
                else:
                    st.warning(f"Could not display any of the {len(all_items)} items (check permissions)")
                
                st.markdown("---")
                
                if is_zarr_store(current_path):
                    if st.button("Select Current Folder", key=f'{key_prefix}_select_current', type="primary"):
                        st.session_state[pending_key] = current_path
                        st.session_state[browse_key] = False
                        st.rerun()
                        
        except PermissionError:
            st.error(f"Permission denied to access: {current_path}")
        except Exception as e:
            st.error(f"Error browsing: {e}")

    return input_path
