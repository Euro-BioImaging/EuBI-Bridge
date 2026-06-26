# EuBI-Bridge 0.1.2b8 - CLI + GUI container
#
# GUI runs via a built-in VNC server - no X server needed on the host.
# Access the GUI by opening http://localhost:6080/vnc.html in any browser.
#
# Build:
#   docker build -t eubi-bridge:0.1.2b8 .
#
# Run GUI (Windows / macOS / Linux):
#   docker run --rm -p 6080:6080 -v /path/to/data:/data eubi-bridge:0.1.2b8 eubi-gui
#   -> open http://localhost:6080/vnc.html
#   -> your files are accessible at /data inside the GUI
#
# Run CLI:
#   docker run --rm -v /path/to/data:/data eubi-bridge:0.1.2b8 eubi to_zarr /data/input.tif /data/output.zarr
#
# HPC (Apptainer) with native display:
#   apptainer exec --env DISPLAY=$DISPLAY -B /path/to/data:/data eubi-bridge.sif eubi-gui
#
# HPC (Apptainer) with built-in VNC:
#   apptainer exec -B /path/to/data:/data eubi-bridge.sif eubi-gui
#   -> open http://localhost:6080/vnc.html

FROM python:3.12-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ARG EUBI_VERSION=0.1.2b8

LABEL org.opencontainers.image.title="EuBI-Bridge" \
      org.opencontainers.image.version="${EUBI_VERSION}" \
      org.opencontainers.image.description="OME-Zarr conversion tool with GUI and CLI" \
      org.opencontainers.image.url="https://github.com/Euro-BioImaging/EuBI-Bridge"

# -- System dependencies -------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Virtual framebuffer + VNC server + noVNC web client
    xvfb \
    x11vnc \
    novnc \
    # Qt6 xcb platform plugin dependencies
    libxcb-cursor0 \
    libxcb1 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    libxcb-util1 \
    # OpenGL / EGL
    libgl1 \
    libegl1 \
    libgles2 \
    # X11 base libraries
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxfixes3 \
    libxcursor1 \
    libxi6 \
    libxtst6 \
    # Font rendering
    libfontconfig1 \
    libfreetype6 \
    # GLib and D-Bus
    libglib2.0-0 \
    libdbus-1-3 \
    # OpenMP for C++ extensions (CZI support)
    libgomp1 \
    # Compression codecs
    zlib1g \
    # HTTPS support
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# -- Python packages -----------------------------------------------------------
# Install from local source to pick up unreleased changes.
# Switch back to PyPI once 0.1.2b8 is published:
#   RUN pip install --no-cache-dir --pre "eubi-bridge==${EUBI_VERSION}"
COPY . /src
RUN pip install --no-cache-dir /src

# -- Qt / display environment --------------------------------------------------
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/PyQt6/Qt6/lib
ENV QT_QPA_PLATFORM=xcb

# -- Entrypoint: intercepts eubi-gui to start VNC; passes everything else through
RUN printf '#!/bin/bash\n\
\n\
if [ "$1" = "eubi-gui" ]; then\n\
    DISPLAY_NUM=${DISPLAY_NUM:-99}\n\
    VNC_PORT=${VNC_PORT:-5900}\n\
    NOVNC_PORT=${NOVNC_PORT:-6080}\n\
    RESOLUTION=${RESOLUTION:-1920x1080x24}\n\
\n\
    # Only start VNC if no display is already available\n\
    if [ -z "$DISPLAY" ]; then\n\
        export DISPLAY=:${DISPLAY_NUM}\n\
        Xvfb :${DISPLAY_NUM} -screen 0 ${RESOLUTION} &\n\
        sleep 0.5\n\
        x11vnc -display :${DISPLAY_NUM} -forever -nopw -shared \\\n\
               -rfbport ${VNC_PORT} -quiet &\n\
        sleep 0.3\n\
        websockify --web /usr/share/novnc ${NOVNC_PORT} localhost:${VNC_PORT} &\n\
        echo ""\n\
        echo "  EuBI-Bridge GUI -> http://localhost:${NOVNC_PORT}/vnc.html"\n\
        echo ""\n\
    fi\n\
fi\n\
\n\
exec "$@"\n' > /usr/local/bin/docker-entrypoint.sh \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 6080

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["eubi", "--help"]
