---
title: EuBI-Bridge - OME-Zarr Conversion Tool
---

# EuBI-Bridge

EuBI-Bridge is a tool for distributed conversion of microscopic image collections to the OME-Zarr format (**v0.4 or v0.5**). 
It can run on the command line or as part of a Python script.  

A key feature of EuBI-Bridge is **aggregative conversion**, which concatenates multiple images along specified dimensions—particularly 
useful for handling large datasets stored as TIFF file collections.  

EuBI-Bridge is built on several powerful libraries, including `zarr`, `bioio`, `dask-distributed` and `tensorstore`, among others. 
Relying on `bioio` plugins for reading, EuBI-Bridge supports a wide range of input file formats. 


---

## Key Features

- Parallelised batch conversion to OME-Zarr version **0.4 or 0.5 (with sharding)**
- Conversion with multi-dimensional concatenation
- Distributed conversion on HPC clusters
- N-dimensional chunking/sharding
- N-dimensional downscaling
- Options for displaying/updating pixel metadata

---

<h2>Installation</h2>

<p>The following steps can be followed to install EuBI-Bridge:</p>

<ol>
  <li>
    <p><strong>Create a conda environment with the required dependencies:</strong></p>
    <pre><code class="language-bash">mamba create -n eubizarr openjdk=11.* maven python=3.12</code></pre>
    <blockquote>
      <strong>ℹ️ Specify either python=3.11 or python=3.12.
      EuBI-Bridge is currently only compatible with Python 3.11 or 3.12 due to conflicting dependencies. We are working on supporting a wider range of Python versions in future releases.</strong>
    </blockquote>
  </li>
  <li>
    <p><strong>Activate the environment and install EuBI-Bridge via pip:</strong></p>
    <pre><code class="language-bash">conda activate eubizarr
pip install --no-cache-dir eubi-bridge==0.1.2</code></pre>
  </li>
</ol>

<blockquote>
  <strong>ℹ️ Linux users:</strong> the graphical interface needs Qt's system
  libraries, which <code>pip</code> does not install. If <code>eubi-gui</code>
  stops with <code>libEGL.so.1: cannot open shared object file</code>, install
  them with
  <code>sudo apt install libegl1 libgl1 libxkbcommon-x11-0 libdbus-1-3 libxcb-cursor0</code>
  (Debian/Ubuntu), or add <code>pyqt6</code> to the <code>mamba create</code>
  command above so conda supplies them. The <code>eubi</code> command-line
  interface works without them.
</blockquote>
<hr>


## Additional Notes

- EuBI-Bridge is in the **beta stage**, and significant updates may be expected.
- **Community support:** Questions and contributions are welcome! Please report any issues.

