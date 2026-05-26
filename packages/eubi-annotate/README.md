# eubi-annotate

Pixel annotation and ML classifier for [EuBI-Bridge](https://github.com/Euro-BioImaging/EuBI-Bridge).

Provides interactive brush-based pixel annotation on OME-Zarr datasets and supervised classification using Random Forest and XGBoost classifiers. Results are written as NGFF-compliant label layers.

## Installation

```bash
pip install eubi-annotate
```

Or via the EuBI-Bridge extra:

```bash
pip install "eubi-bridge[annotate]"
```

## Features

- Interactive paint-brush annotation on 2D slices
- Feature extraction using Gaussian derivatives and rank filters
- Random Forest and XGBoost classifiers via scikit-learn
- Output written as OME-NGFF v0.4 label layers

## Requirements

- `eubi-bridge >= 0.1.2b1`
- `scikit-learn >= 1.0`
- `xgboost >= 1.7`

For deep learning features (DINOv2 embeddings), install `eubi-deep` as well.
