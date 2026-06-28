# Tutorial

Welcome to the EuBI-Bridge conversion tutorial. Here we demonstrate how to convert batches
of image datasets to OME-Zarr using the EuBI-Bridge CLI. 

EuBI-Bridge supports two different conversion modes: **unary** (one-to-one) and **aggregative** (multiple-to-one) conversion. Unary conversion converts each input file to a single OME-Zarr container, whereas aggregative conversion concatenates input images along specified dimensions. Below we explain each of these modes with examples.

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

By default EuBI-Bridge writes OME-Zarr **0.4** (backed by Zarr v2). To create OME-Zarr
**0.5** (backed by Zarr v3, which enables sharding), add the `--ome_zarr_version 0.5` argument:

```bash
eubi to_zarr multichannel_timeseries multichannel_timeseries_zarr --ome_zarr_version 0.5
```  

This produces:  

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

This produces:

```bash
multichannel_timeseries_zarr
├── Channel1-T0001.zarr
├── Channel1-T0002.zarr
├── Channel1-T0003.zarr
└── Channel1-T0004.zarr
```
### Aggregative Conversion (Concatenation Along Dimensions)  

To concatenate images along specific dimensions, EuBI-Bridge needs to be informed
of file patterns that specify image dimensions. For this example,
the file pattern for the channel dimension is `Channel`, which is followed by the channel index,
and the file pattern for the time dimension is `T`, which is followed by the time index.

To concatenate along the **time** dimension:

```bash
eubi to_zarr multichannel_timeseries multichannel_timeseries_concat_zarr --channel_tag Channel --time_tag T --concatenation_axes t
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
eubi to_zarr multichannel_timeseries multichannel_timeseries_concat_zarr --channel_tag Channel --time_tag T --concatenation_axes ct
```  

Note that both axes are specified via the argument `--concatenation_axes ct`.

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
eubi to_zarr multichannel_timeseries_nested multichannel_timeseries_nested_concat_zarr --channel_tag Channel --time_tag T --concatenation_axes ct
```  

Output:  

```bash
multichannel_timeseries_nested_concat_zarr
└── Channel_cset-T_tset.zarr
```  

To concatenate along the channel dimension only:  

```bash
eubi to_zarr multichannel_timeseries_nested multichannel_timeseries_nested_concat_zarr --channel_tag Channel --time_tag T --concatenation_axes c
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
eubi to_zarr "multichannel_timeseries_nested/**/*T0003*" multichannel_timeseries_nested_concat_zarr --channel_tag Channel --time_tag T --concatenation_axes c
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
eubi to_zarr blueredchannels_timeseries blueredchannels_timeseries_concat_zarr --channel_tag Blue,Red --time_tag T --concatenation_axes ct
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
│   ├── T0004.tif
├── Red
│   ├── T0001.tif
│   ├── T0002.tif
│   ├── T0003.tif
│   ├── T0004.tif
```  


One can run the exact same command:

```bash
eubi to_zarr blueredchannels_timeseries_nested blueredchannels_timeseries_nested_concat_zarr --channel_tag Blue,Red --time_tag T --concatenation_axes ct
```  

Output:  

```bash
blueredchannels_timeseries_nested_concat_zarr
└── BlueRed_cset-T_tset.zarr
```


