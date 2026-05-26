# eubi-deep

Deep learning features for [EuBI-Bridge](https://github.com/Euro-BioImaging/EuBI-Bridge) annotation.

Extends `eubi-annotate` with DINOv2-based feature extraction, enabling self-supervised deep features as input to the pixel classifier.

## Installation

```bash
pip install eubi-deep
```

Or via the EuBI-Bridge extra:

```bash
pip install "eubi-bridge[deep]"
```

## Requirements

- `eubi-annotate >= 0.1.2b1`
- `torch >= 2.0`

PyTorch must be installed separately to match your local CUDA version:

```bash
# CPU-only
pip install torch

# CUDA 12.x
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
