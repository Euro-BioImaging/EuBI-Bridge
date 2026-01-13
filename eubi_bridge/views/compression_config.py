"""
Streamlit component for unified compression configuration.

Provides a dynamic UI for selecting and configuring compressors
compatible with both Zarr v2 and v3.
"""

import streamlit as st
from eubi_bridge.external.dyna_zarr.codecs import Codecs


def render_compression_config(key_prefix=""):
    """
    Render a compression configuration panel in Streamlit.
    
    Displays a compressor selector with dynamic parameters based on
    the chosen compression algorithm.
    
    Parameters
    ----------
    key_prefix : str, optional
        Prefix for Streamlit widget keys to avoid conflicts.
        
    Returns
    -------
    tuple
        (compressor_name: str, compressor_params: dict)
        E.g., ('gzip', {'level': 5}) or ('blosc', {'clevel': 5, 'cname': 'lz4', 'shuffle': 1})
    """
    with st.expander("🗜️ Compression Settings", expanded=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            compressor = st.selectbox(
                "Compression Algorithm",
                options=['blosc', 'zstd', 'gzip', 'lz4', 'bz2', 'none'],
                index=0,
                help="Choose compression algorithm. 'blosc' is fastest, 'zstd' offers best compression, 'none' disables compression.",
                key=f"{key_prefix}_compressor"
            )
        
        with col2:
            st.write("")  # Spacing
        
        # Dynamic parameters based on selected compressor
        compressor_params = {}
        
        if compressor == 'blosc':
            st.markdown("**Blosc Parameters:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                clevel = st.slider(
                    "Compression Level",
                    min_value=1,
                    max_value=9,
                    value=5,
                    help="Higher = more compression but slower",
                    key=f"{key_prefix}_blosc_clevel"
                )
                compressor_params['clevel'] = clevel
            
            with col2:
                cname = st.selectbox(
                    "Blosc Codec",
                    options=['lz4', 'zstd', 'zlib', 'snappy', 'blosclz'],
                    index=0,
                    help="Inner codec for Blosc. LZ4=fast, ZSTD=better compression",
                    key=f"{key_prefix}_blosc_cname"
                )
                compressor_params['cname'] = cname
            
            with col3:
                shuffle = st.selectbox(
                    "Shuffle Mode",
                    options=['no shuffle', 'byte shuffle', 'bit shuffle'],
                    index=1,  # byte shuffle is default
                    help="Data reordering before compression",
                    key=f"{key_prefix}_blosc_shuffle"
                )
                # Convert to numeric value (0, 1, 2)
                shuffle_map = {'no shuffle': 0, 'byte shuffle': 1, 'bit shuffle': 2}
                compressor_params['shuffle'] = shuffle_map[shuffle]
        
        elif compressor == 'zstd':
            st.markdown("**ZSTD Parameters:**")
            col1, col2 = st.columns([2, 2])
            
            with col1:
                clevel = st.slider(
                    "Compression Level",
                    min_value=1,
                    max_value=22,
                    value=3,
                    help="1-3=fast, 10-20=slower but better compression, 22=ultra",
                    key=f"{key_prefix}_zstd_clevel"
                )
                compressor_params['clevel'] = clevel
            
            with col2:
                st.info("💡 ZSTD level 10-15 offers good balance")
        
        elif compressor == 'gzip':
            st.markdown("**GZip Parameters:**")
            col1, col2 = st.columns([2, 2])
            
            with col1:
                level = st.slider(
                    "Compression Level",
                    min_value=1,
                    max_value=9,
                    value=5,
                    help="1=fastest, 9=best compression",
                    key=f"{key_prefix}_gzip_level"
                )
                compressor_params['clevel'] = level
            
            with col2:
                st.info("💡 GZip is widely compatible but slower than modern codecs")
        
        elif compressor == 'lz4':
            st.markdown("**LZ4 Parameters:**")
            st.info("LZ4 uses default parameters (no configuration needed)")
            # LZ4 has no parameters in our implementation
            compressor_params = {}
        
        elif compressor == 'bz2':
            st.markdown("**BZ2 Parameters:**")
            col1, col2 = st.columns([2, 2])
            
            with col1:
                level = st.slider(
                    "Compression Level",
                    min_value=1,
                    max_value=9,
                    value=5,
                    help="1=fastest, 9=best compression",
                    key=f"{key_prefix}_bz2_level"
                )
                compressor_params['clevel'] = level
            
            with col2:
                st.info("💡 BZ2 is slow, prefer ZSTD or Blosc")
        
        elif compressor == 'none':
            st.warning("⚠️ No compression enabled")
            compressor_params = {}
        
        # Show preview of configuration
        st.divider()
        st.subheader("Configuration Preview")
        
        # Create a temporary Codecs instance to show what will be generated
        codecs = Codecs(compressor=compressor if compressor != 'none' else None, **compressor_params)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Zarr v2 Format:**")
            v2_config = codecs.to_v2_config()
            if v2_config:
                # Note: TensorStore only uses 'id', parameters handled by numcodecs
                st.json({"id": v2_config.get("id", "none")})
                if compressor_params:
                    st.caption("⚠️ Parameters applied by numcodecs backend, not in TensorStore spec")
            else:
                st.info("No compression")
        
        with col2:
            st.write("**Zarr v3 Format:**")
            v3_config = codecs.to_v3_config()
            st.json(v3_config)
        
        # Return compressor name and params
        if compressor == 'none':
            return None, {}
        
        # Handle clevel vs level parameter naming
        if 'clevel' in compressor_params and compressor in ('gzip', 'bz2'):
            compressor_params['level'] = compressor_params.pop('clevel')
        
        return compressor, compressor_params
