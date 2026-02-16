# EuBI-Bridge  

[![Documentation](https://img.shields.io/badge/documentation-online-green)](https://euro-bioimaging.github.io/EuBI-Bridge/)

EuBI-Bridge is a tool for distributed conversion of microscopic image collections into the OME-Zarr format. 
It can run on the command line or as part of a Python script.  

A key feature of EuBI-Bridge is **aggregative conversion**, which concatenates multiple images along specified dimensions—particularly useful for handling large datasets stored as TIFF file collections.  

EuBI-Bridge is built on several powerful libraries, including `zarr`, `bioio`, `dask` and `tensorstore`, among others. 

Relying on `bioio` plugins for reading, EuBI-Bridge supports a wide range of input file formats. 


## Installation

The recommended way to install EuBI-Bridge is via pip. Create a virtual environment with **Python 3.11 or 3.12** and use pip to install EuBI-Bridge as shown below:

```bash
python -m venv venv # Python must be either version 3.11 or 3.12.
source venv/bin/activate
pip install 'eubi-bridge[all]==0.1.0c4' # installs both GUI and CLI
# OR
# pip install 'eubi-bridge[cli]==0.1.0c4' # installs only CLI
# pip install 'eubi-bridge[gui]==0.1.0c4' # installs only GUI
# pip install 'eubi-bridge==0.1.0c4' # installs as a Python library, without GUI or CLI utilities.
#
# If a previous version of eubi-bridge was installed before, reset the configuration:
eubi reset_config
```

**Important: EuBI-Bridge is currently only compatible with Python 3.11 or 3.12 due to conflicting dependencies. 
We are working on supporting a wider range of Python versions in future releases.**

If your default Python is different from version 3.11 or 3.12, create a conda environment with one of these
Python versions:

```bash
mamba create -n eubizarr python=3.12
```

Then install EuBI-Bridge via pip in the conda environment:

```bash
conda activate eubizarr
pip install --no-cache-dir 'eubi-bridge[all]==0.1.0c4'
# If a previous version of eubi-bridge was installed before, reset the configuration:
eubi reset_config
```

#### Troubleshooting

If you receive a `Building wheel` error such as:

```bash
  Building wheel for ... error
  error: subprocess-exited-with-error
  
  × python setup.py bdist_wheel did not run successfully.
  │ exit code: 1
```
then try the following:

```bash
# In the `eubizarr` environment
mamba install cmake zlib boost # preinstall dependencies that can help build from source
pip install --no-cache-dir "eubi-bridge[all]==0.1.0c4" # try installing again with the dependencies available
# If a previous version of eubi-bridge was installed before, reset the configuration:
eubi reset_config
````

## Documentation

Find the documentation for EuBI-Bridge [here](https://euro-bioimaging.github.io/EuBI-Bridge/)

## Basic Usage  

### Unary Conversion  

Given a dataset structured as follows: 

```bash
multichannel_timeseries
├── Channel1-T0001.tif
├── Channel1-T0002.tif
├── Channel1-T0003.tif
├── Channel1-T0004.tif
├── Channel2-T0001.tif
├── Channel2-T0002.tif
├── Channel2-T0003.tif
└── Channel2-T0004.tif
```  

To convert each TIFF into a separate OME-Zarr container (unary conversion):  

```bash
eubi to_zarr multichannel_timeseries multichannel_timeseries_zarr
```  

Use the argument `--zarr_format` to specify the zarr format version to use.
To create a zarr version 3 dataset, use `--zarr_format 3`:

```bash
eubi to_zarr multichannel_timeseries multichannel_timeseries_zarr --zarr_format 3
```  

Both of these commands will perform unary conversion, resulting in the following output:  

```bash
multichannel_timeseries_zarr
├── Channel1-T0001.zarr
├── Channel1-T0002.zarr
├── Channel1-T0003.zarr
├── Channel1-T0004.zarr
├── Channel2-T0001.zarr
├── Channel2-T0002.zarr
├── Channel2-T0003.zarr
└── Channel2-T0004.zarr
```  

Use **wildcards** to specifically convert the images belonging to Channel1:

```bash
eubi to_zarr "multichannel_timeseries/Channel1*" multichannel_timeseries_channel1_zarr
```

### Aggregative Conversion (Concatenation Along Dimensions)  

To concatenate images along specific dimensions, EuBI-Bridge needs to be informed
of file patterns that specify image dimensions. For this example,
the file pattern for the channel dimension is `Channel`, which is followed by the channel index,
and the file pattern for the time dimension is `T`, which is followed by the time index.

To concatenate along the **time** dimension:

```bash
eubi to_zarr multichannel_timeseries multichannel_timeseries_concat_zarr \
--channel_tag Channel \
--time_tag T \
--concatenation_axes t
```  

Output:  

```bash
multichannel_timeseries_time-concat_zarr
├── Channel1-T_tset.zarr
└── Channel2-T_tset.zarr
```  

**Important note:** if the `--channel_tag` was not provided, the tool would not be aware
of the multiple channels in the image and try to concatenate all images into a single one-channeled OME-Zarr. Therefore, 
when an aggregative conversion is performed, all dimensions existing in the input files must be specified via their respective tags. 

For multidimensional concatenation (**channel** + **time**):

```bash
eubi to_zarr multichannel_timeseries multichannel_timeseries_concat_zarr \
--channel_tag Channel \
--time_tag T \
--concatenation_axes ct
```  

Note that both axes are specified wia the argument `--concatenation_axes ct`.

Output:

```bash
multichannel_timeseries_concat_zarr
└── Channel_cset-T_tset.zarr
```  

### Handling Nested Directories  

For datasets stored in nested directories such as:  

```bash
multichannel_timeseries_nested
├── Channel1
│   ├── T0001.tif
│   ├── T0002.tif
│   ├── T0003.tif
│   ├── T0004.tif
├── Channel2
│   ├── T0001.tif
│   ├── T0002.tif
│   ├── T0003.tif
│   ├── T0004.tif
```  

EuBI-Bridge automatically detects the nested structure. To concatenate along both channel and time dimensions:  

```bash
eubi to_zarr \
multichannel_timeseries_nested \
multichannel_timeseries_nested_concat_zarr \
--channel_tag Channel \
--time_tag T \
--concatenation_axes ct
```  

Output:  

```bash
multichannel_timeseries_nested_concat_zarr
└── Channel_cset-T_tset.zarr
```  

To concatenate along the channel dimension only:  

```bash
eubi to_zarr \
multichannel_timeseries_nested \
multichannel_timeseries_nested_concat_zarr \
--channel_tag Channel \
--time_tag T \
--concatenation_axes c
```  

Output:  

```bash
multichannel_timeseries_nested_concat_zarr
├── Channel_cset-T0001.zarr
├── Channel_cset-T0002.zarr
├── Channel_cset-T0003.zarr
└── Channel_cset-T0004.zarr
```  

### Selective Data Conversion    

To recursively select specific files for conversion, wildcard patterns can be used. 
For example, to concatenate only **timepoint 3** along the channel dimension:  

```bash
eubi to_zarr \
"multichannel_timeseries_nested/**/*T0003*" \
multichannel_timeseries_nested_concat_zarr \
--channel_tag Channel \
--time_tag T \
--concatenation_axes c
```  

Output:  

```bash
multichannel_timeseries_nested_concat_zarr
└── Channel_cset-T0003.zarr
```  

**Note:** When using wildcards, the input directory path must be enclosed 
in quotes as shown in the example above.  

### Handling Categorical Dimension Patterns  

For datasets where channel names are categorical such as in:

```bash
blueredchannel_timeseries
├── Blue-T0001.tif
├── Blue-T0002.tif
├── Blue-T0003.tif
├── Blue-T0004.tif
├── Red-T0001.tif
├── Red-T0002.tif
├── Red-T0003.tif
└── Red-T0004.tif
```

Specify categorical names as a comma-separated list:  

```bash
eubi to_zarr \
blueredchannels_timeseries \
blueredchannels_timeseries_concat_zarr \
--channel_tag Blue,Red \
--time_tag T \
--concatenation_axes ct
```  

Output:  

```bash
blueredchannels_timeseries_concat_zarr
└── BlueRed_cset-T_tset.zarr
```  

Note that the categorical names are aggregated in the output OME-Zarr name.  


With nested input structure such as in:  

```bash
blueredchannels_timeseries_nested
├── Blue
│   ├── T0001.tif
│   ├── T0002.tif
│   ├── T0003.tif
│   └── T0004.tif
└── Red
    ├── T0001.tif
    ├── T0002.tif
    ├── T0003.tif
    └── T0004.tif
```  

One can run the exact same command:

```bash
eubi to_zarr \
blueredchannels_timeseries_nested \
blueredchannels_timeseries_nested_concat_zarr \
--channel_tag Blue,Red \
--time_tag T \
--concatenation_axes ct
```  

Output:  

```bash
blueredchannels_timeseries_nested_concat_zarr
└── BlueRed_cset-T_tset.zarr
```

## Additional Notes

- EuBI-Bridge is in the **beta stage**, and significant updates may be expected.
- **Community support:** Questions and contributions are welcome! Please report any issues.


