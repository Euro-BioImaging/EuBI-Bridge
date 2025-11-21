FROM condaforge/mambaforge:latest

# EuBI-Bridge Docker Container

WORKDIR /workspace

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd -g ${GROUP_ID} eubi && \
    useradd -m -u ${USER_ID} -g eubi -s /bin/bash eubi

# Create the env with java dependencies
RUN mamba create -n eubizarr openjdk=11.* maven python=3.12 -y && \
    mamba clean -afy

# Activate the environment and install eubi-bridge
SHELL ["/bin/bash", "-c"]
RUN source activate eubizarr && \
    pip install --no-cache-dir git+https://github.com/Euro-BioImaging/EuBI-Bridge.git@async && \
    eubi reset_config

# CRITICAL: Pre-initialize JGO and Maven dependencies during build
# This ensures all Maven artifacts are downloaded into the image

RUN source activate eubizarr && \
    echo "Pre-downloading Maven/JGO dependencies (this will take several minutes)..." && \
    mkdir -p /tmp/dummy_input && \
    echo "Creating a minimal test file to trigger dependency downloads..." && \
    python -c "import numpy as np; from PIL import Image; img = Image.fromarray(np.random.randint(0, 255, (64, 64), dtype=np.uint8)); img.save('/tmp/dummy_input/test.tif')" && \
    (timeout 600 eubi to_zarr /tmp/dummy_input/test.tif /tmp/dummy_output 2>&1 || true) && \
    rm -rf /tmp/dummy_input /tmp/dummy_output && \
    echo "Maven/JGO dependencies fully cached"

# Store pre-populated config in /opt for easy extraction
RUN mkdir -p /opt/eubi-config/.eubi_bridge /opt/eubi-config/.jgo /opt/eubi-config/.cache && \
    cp -r /root/.eubi_bridge/* /opt/eubi-config/.eubi_bridge/ 2>/dev/null || true && \
    cp -r /root/.jgo/* /opt/eubi-config/.jgo/ 2>/dev/null || true && \
    cp -r /root/.cache/* /opt/eubi-config/.cache/ 2>/dev/null || true && \
    chmod -R 777 /opt/eubi-config

# Also copy to default user home for fallback (slower, no volume)
RUN mkdir -p /home/eubi/.eubi_bridge /home/eubi/.jgo /home/eubi/.cache && \
    cp -r /opt/eubi-config/.eubi_bridge/* /home/eubi/.eubi_bridge/ 2>/dev/null || true && \
    cp -r /opt/eubi-config/.jgo/* /home/eubi/.jgo/ 2>/dev/null || true && \
    cp -r /opt/eubi-config/.cache/* /home/eubi/.cache/ 2>/dev/null || true && \
    chown -R eubi:eubi /home/eubi && \
    chmod -R 755 /home/eubi

RUN chown -R eubi:eubi /workspace

USER eubi

# Make sure conda environment is activated automatically
ENV PATH="/opt/conda/envs/eubizarr/bin:${PATH}"
ENV CONDA_DEFAULT_ENV=eubizarr
ENV CONDA_PREFIX=/opt/conda/envs/eubizarr
ENV HOME=/home/eubi

# This is supposed to speed up the container with java code:
ENV JAVA_TOOL_OPTIONS="-XX:+UseContainerSupport -XX:MaxRAMPercentage=75.0 -XX:InitialRAMPercentage=50.0"

ENTRYPOINT ["eubi"]
CMD ["--help"]
