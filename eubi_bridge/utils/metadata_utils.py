from eubi_bridge.utils.logging_config import get_logger

logger = get_logger(__name__)

def get_printables(
                   axes: str,
                   shapedict: dict,
                   scaledict: dict,
                   unitdict: dict
                   ):
    dimensions = [dim for dim in axes if dim in scaledict.keys()]

    rows = [("Dimension", "Size (pixels)", "Scale", "Unit")]
    for i, dim in enumerate(dimensions):
        # size = shape[i] if i < len(shape) else ''
        size = shapedict.get(dim, '')
        scale = scaledict.get(dim, '')
        unit = unitdict.get(dim, '')
        rows.append((dim, str(size), str(scale), unit))

    col_widths = [max(len(str(row[i])) for row in rows) for i in range(4)]

    printables = []

    # ANSI escape codes for bold
    BOLD = '\033[1m'
    RESET = '\033[0m'

    for idx, row in enumerate(rows):
        line = "  ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
        if idx == 0:
            line = BOLD + line + RESET
        printables.append(line)

    return printables

def print_printable(printable):
    for item in printable:
        print(item)

# def show_pixel_meta(input_path):
#     # input_path = f"/home/oezdemir/PycharmProjects/dask_env1/data/tifflist"
#     base = BridgeBase(input_path)
#     base.read_dataset(True)
#     base.digest()
#     base.compute_pixel_metadata()
#     ###
#     printables = {}
#     for path, vmeta in base.pixel_metadata.vmetaset.items():
#         shape = vmeta.shape
#         scaledict = vmeta.scaledict
#         unitdict = vmeta.unitdict
#         printable = get_printables(shape,scaledict,unitdict)
#         printables[path] = printable
#     for path, printable in printables.items():
#         print('---------')
#         print(f"")
#         print(f"Metadata for '{path}':")
#         print_printable(printable)


def generate_channel_metadata(num_channels,
                              dtype='np.uint16'
                              ):
    # Standard distinct microscopy colors
    default_colors = [
        "FF0000",  # Red
        "00FF00",  # Green
        "0000FF",  # Blue
        "FF00FF",  # Magenta
        "00FFFF",  # Cyan
        "FFFF00",  # Yellow
        "FFFFFF",  # White
    ]

    channels = []
    import numpy as np

    if dtype is not None and np.issubdtype(dtype, np.integer):
        min, max = np.iinfo(dtype).min, np.iinfo(dtype).max
    elif dtype is not None and np.issubdtype(dtype, np.floating):
        min, max = np.finfo(dtype).min, np.finfo(dtype).max
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    for i in range(num_channels):
        color = default_colors[i] if i < len(
            default_colors) else f"{i * 40 % 256:02X}{i * 85 % 256:02X}{i * 130 % 256:02X}"
        channel = {
            "color": color,
            "coefficient": 1,
            "active": True,
            "label": f"Channel {i}",
            "window": {
                "min": min,
                "max": max,
                "start": min,
                "end": max
            },
            "family": "linear",
            "inverted": False
        }
        channels.append(channel)

    return channels


class ChannelMap:
    DEFAULT_COLORS = {
        'red': "FF0000",
        'green': "00FF00",
        'blue': "0000FF",
        'magenta': "FF00FF",
        'cyan': "00FFFF",
        'yellow': "FFFF00",
        'white': "FFFFFF",
    }
    def __getitem__(self, key):
        if key in self.DEFAULT_COLORS:
            return self.DEFAULT_COLORS[key]
        else:
            return None


def parse_channels(manager,
                   **kwargs
                   ):

    dtype = kwargs.get('dtype', None)
    if dtype is None:
        dtype = manager.array.dtype
    if 'c' not in manager.axes:
        channel_count = 1
    else:
        channel_idx = manager.axes.index('c')
        channel_count = manager.array.shape[channel_idx]
        assert channel_count == len(manager.channels), f"Manager constructed incorrectly!"
    default_channels = generate_channel_metadata(num_channels=channel_count,
                                                 dtype=dtype)

    if manager.channels is not None:
        for idx, channel in enumerate(manager.channels):
            default_channels[idx].update(channel)

    import copy
    from eubi_bridge.utils.convenience import make_json_safe
    output = copy.deepcopy(default_channels)
    assert 'coefficient' in output[0].keys(), f"Channels parsed incorrectly!" # A very basic validation

    #######-----------------------------------##################
    # Handle channel intensity limits first
    channel_intensity_limits = kwargs.get('channel_intensity_limits','from_dtype')
    assert channel_intensity_limits in ('from_dtype', 'from_array'), f"Channel intensity limits must be either 'from_dtype' or 'from_array'"
    from_array = channel_intensity_limits == 'from_array'
    start_intensities, end_intensities = manager.compute_intensity_limits(
                                                    from_array = from_array,
                                                    dtype = dtype)
    mins, maxes = manager.compute_intensity_extrema(dtype = dtype)
    # The channel intensity window not controlled by the channel_indices parameter.
    # Parse both channel intensity and hexadecimal color code for all channels:
    for channel_idx in range(len(output)):
        current_channel = output[channel_idx]
        color = current_channel['color']
        if color.startswith('#'):
            color = color[1:]
        if len(color) == 6:
            pass
        elif len(color) == 12:
            logger.warn(f"The color code is being parsed from 12- to 6-hex format.")
            print(f"The color code is being parsed from 12- {color} to 6-hex {color[::2]} format.")
            color = color[::2]
        else:
            logger.warn(f"The color code does not follow a hex format."
                        f"The color code is truncated to the first 6 characters." )
            print(f"The color code is truncated to the first 6 characters: {color[:6]}.")
            color = color[:6]
        current_channel['color'] = color
        window = {
            'min': mins[channel_idx],
            'max': maxes[channel_idx],
            'start': start_intensities[channel_idx],
            'end': end_intensities[channel_idx]
        }
        current_channel['window'] = window
        output[channel_idx] = current_channel
    #######-----------------------------------##################

    channel_indices = kwargs.get('channel_indices', [])

    if channel_indices == 'all':
        channel_indices = list(range(channel_count))
    if not hasattr(channel_indices, '__len__'):
        channel_indices = [channel_indices]
    channel_labels = kwargs.get('channel_labels', None)
    channel_colors = kwargs.get('channel_colors', None)
    if channel_labels in ('auto', None):
        channel_labels = [channel_labels] * len(channel_indices)
    if channel_colors in ('auto', None):
        channel_colors = [channel_colors] * len(channel_indices)

    import numpy as np
    try:
        if np.isnan(channel_indices):
            return output
        elif channel_indices is None:
            return output
        elif channel_indices == []:
            return output
    except:
        pass
    if isinstance(channel_indices, str):
        channel_indices = [i for i in channel_indices.split(',')]
    if isinstance(channel_labels, str):
        channel_labels = [i for i in channel_labels.split(',')]
    if isinstance(channel_colors, str):
        channel_colors = [i for i in channel_colors.split(',')]

    channel_indices = [int(i) for i in channel_indices]
    items = [channel_indices, channel_labels, channel_colors, channel_intensity_limits]
    for idx, item in enumerate(items):
        if not isinstance(item, str) and np.isscalar(item):
            item = [item]
        for i, it in enumerate(item):
            try:
                if np.isnan(it):
                    item[i] = 'auto'
            except:
                pass
        items[idx] = item
    channel_indices, channel_labels, channel_colors, channel_intensity_limits = items
    import pprint
    pprint.pprint(manager.path)
    pprint.pprint(manager.series_path)
    pprint.pprint(manager.array.shape)
    pprint.pprint(channel_indices)
    pprint.pprint(channel_labels)
    pprint.pprint(channel_colors)
    if len(channel_indices) > channel_count:
        pprint.pprint(f"For the path {manager.series_path} and array {manager.array.shape}:")
        pprint.pprint(f"Channel indices wrongly specified as {channel_indices}. Being corrected to {channel_count}")
        channel_indices = channel_indices[:channel_count]
    if len(channel_labels) > channel_count:
        pprint.pprint(f"For the path {manager.series_path} and array {manager.array.shape}:")
        pprint.pprint(f"Channel labels wrongly specified as {channel_labels}. Being corrected to {channel_count}")
        channel_labels = channel_labels[:channel_count]
    if len(channel_colors) > channel_count:
        pprint.pprint(f"For the path {manager.series_path} and array {manager.array.shape}:")
        pprint.pprint(f"Channel colors wrongly specified as {channel_colors}. Being corrected to {channel_count}")
        channel_colors = channel_colors[:channel_count]


    if not len(channel_indices) == len(channel_labels) == len(channel_colors):
        raise ValueError(f"Channel indices, labels, colors, intensity minima and extrema must have the same length. \n"
                         f"Currently they are {channel_indices},{channel_labels},{channel_colors} \n"
                         f"So you need to specify --channel_indices, --channel_labels, --channel_colors, --channel_intensity_extrema with the same number of elements. \n"
                         f"To keep specific labels or colors unchanged, add 'auto'. E.g. `--channel_indices 0,1 --channel_colors auto,red`")

    cm = ChannelMap()

    if len(channel_indices) == 0:
        return output

    for idx in range(len(channel_indices)):
        channel_idx = channel_indices[idx]
        if channel_idx >= channel_count:
            logger.warn(f"For the dataset {manager.series_path},\n"
                        f"Channel index {channel_idx} is out of range -> {0}:{channel_count - 1}.\n"
                        f"Skipping channel {channel_idx}...")
            continue
        current_channel = output[channel_idx]
        if channel_labels[idx] not in (None, 'auto'):
            current_channel['label'] = channel_labels[idx]
        colorname = channel_colors[idx]
        if colorname not in (None, 'auto'):
            current_channel['color'] = cm[colorname] if cm[colorname] is not None else colorname
         # Add the parameters that are currently hard-coded
        # current_channel['coefficient'] = 1
        # current_channel['active'] = True
        # current_channel['family'] = "linear"
        # current_channel['inverted'] = False
        ###--------------------------------------------------------------------------------###
        output[channel_idx] = current_channel
    ret = make_json_safe(output)
    print(f'channels parsed:{ret}')
    return ret
