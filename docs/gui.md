# Graphical Interface

Launch the graphical interface by running `eubi-gui` in the terminal. The four
short videos below walk you through one complete job: converting several images
to OME-Zarr, and then inspecting one of the outputs.

The videos cover the common scenario: one-to-one conversion from a collection of
files. More advanced features such as aggregative conversions (concatenation of
multiple files), editable batch tables, custom configuration files and editing
output metadata are all supported by the interface but not yet demonstrated in
the videos; demos for those will be provided soon. In the meantime the
[CLI reference](cli_reference/index.md) describes every parameter the interface
exposes, and each control carries a tooltip explaining what it does.

!!! tip "Linux users"

    The graphical interface needs Qt's system libraries, which `pip` does not
    install. If `eubi-gui` stops with `libEGL.so.1: cannot open shared object
    file`, install them with
    `sudo apt install libegl1 libgl1 libxkbcommon-x11-0 libdbus-1-3 libxcb-cursor0`
    on Debian or Ubuntu. The `eubi` command-line interface needs none of them.

## 1. Selecting input and output folders

Navigate to your images, filter them with include and exclude patterns (which
accept star `*` expressions), tick the files to be converted, and choose where
the OME-Zarr output should be written.

<video controls muted playsinline width="100%"
       src="https://github.com/Euro-BioImaging/EuBI-Bridge/releases/download/v0.1.2-media/select-io-folders.mp4">
  Your browser cannot play this video.
  <a href="https://github.com/Euro-BioImaging/EuBI-Bridge/releases/download/v0.1.2-media/select-io-folders.mp4">Download it instead.</a>
</video>

## 2. Choosing conversion parameters

Work through the parameter tabs: reader options, chunking and compression,
downscaling into a resolution pyramid, and the metadata written alongside the
pixels. Each field explains itself on hover, and the settings can be saved to a
configuration file for reuse.

<video controls muted playsinline width="100%"
       src="https://github.com/Euro-BioImaging/EuBI-Bridge/releases/download/v0.1.2-media/parameter-selection.mp4">
  Your browser cannot play this video.
  <a href="https://github.com/Euro-BioImaging/EuBI-Bridge/releases/download/v0.1.2-media/parameter-selection.mp4">Download it instead.</a>
</video>

## 3. Running a conversion

Start the job and watch it progress. The log reports each file as it is read,
converted and downscaled, so a failure points at the file that caused it.

<video controls muted playsinline width="100%"
       src="https://github.com/Euro-BioImaging/EuBI-Bridge/releases/download/v0.1.2-media/run-conversion.mp4">
  Your browser cannot play this video.
  <a href="https://github.com/Euro-BioImaging/EuBI-Bridge/releases/download/v0.1.2-media/run-conversion.mp4">Download it instead.</a>
</video>

## 4. Inspecting the result

Switch to the Inspect tab to open a converted store: axis units and pixel sizes,
the chunk and shard layout of each pyramid level, and the image itself in the
viewer. Pixel sizes can be corrected here and saved back into the OME-Zarr.

<video controls muted playsinline width="100%"
       src="https://github.com/Euro-BioImaging/EuBI-Bridge/releases/download/v0.1.2-media/inspection.mp4">
  Your browser cannot play this video.
  <a href="https://github.com/Euro-BioImaging/EuBI-Bridge/releases/download/v0.1.2-media/inspection.mp4">Download it instead.</a>
</video>

---

The recordings are attached to the
[v0.1.2-media release](https://github.com/Euro-BioImaging/EuBI-Bridge/releases/tag/v0.1.2-media).
They are not part of the repository, so cloning or installing the package does
not download them.
