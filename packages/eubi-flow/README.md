# eubi-flow

Workflow DAG engine and Flow GUI for [EuBI-Bridge](https://github.com/Euro-BioImaging/EuBI-Bridge).

Provides a declarative pipeline system for chaining image processing operations on OME-Zarr datasets, with a visual drag-and-drop editor in the EuBI-Bridge GUI.

## Installation

```bash
pip install eubi-flow
```

Or via the EuBI-Bridge extra:

```bash
pip install "eubi-bridge[flow]"
```

## Usage

```bash
# Create a new flow
eubi flow create myflow --input_path /path/to/data.zarr --workdir /tmp/myflow

# Add processing waves
eubi flow select myflow add_wave gaussian_filter --output_heave blurred --sigma 2.0

# Run the flow
eubi flow select myflow run
```

The Flow tab in the EuBI-Bridge GUI provides a visual canvas for building and running flows interactively.

## Requirements

- `eubi-bridge >= 0.1.2b1`
