import copy

import numpy as np

from eubi_bridge.utils.json_utils import make_json_safe
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


class ChannelParser:
    """Orchestrates channel metadata parsing and customization.
    
    Handles initialization, intensity window application, parameter parsing,
    validation, and application of user customizations to channel metadata.
    """
    
    def __init__(self, manager, **kwargs):
        """Initialize parser with manager and parameters.
        
        Args:
            manager: ArrayManager instance with array and metadata
            **kwargs: Channel customization parameters
        """
        self.manager = manager
        self.kwargs = kwargs
        self.output = None
        self.channel_count = None
        self.channel_intensity_limits = kwargs.get('channel_intensity_limits', 'from_dtype')
    
    def parse(self):
        """Execute full parsing pipeline and return JSON-safe metadata.
        
        Returns:
            List of JSON-safe channel metadata dictionaries
        """
        self._init_channels()
        self._apply_intensity_windows()
        self._apply_user_customizations()
        return make_json_safe(self.output)
    
    def _init_channels(self):
        """Initialize default channels from manager and merge with manager channels."""
        dtype = self.kwargs.get('dtype') or self.manager.array.dtype
        self.channel_count = self._get_channel_count()
        
        default_channels = generate_channel_metadata(
            num_channels=self.channel_count,
            dtype=dtype
        )
        
        self._merge_manager_channels(default_channels)
        self.output = copy.deepcopy(default_channels)
        
        assert 'coefficient' in self.output[0].keys(), \
            "Channels parsed incorrectly!"
    
    def _get_channel_count(self) -> int:
        """Get channel count from manager axes.
        
        Returns:
            Number of channels in the array
        """
        if 'c' not in self.manager.axes:
            return 1
        channel_idx = self.manager.axes.index('c')
        return self.manager.array.shape[channel_idx]
    
    def _merge_manager_channels(self, default_channels: list):
        """Merge manager's channel metadata into defaults.
        
        Args:
            default_channels: List of default channel metadata dictionaries
        """
        if self.manager.channels is None:
            return
        
        # Get channels from pyramid metadata or manager
        mchannels = (
            self.manager.pyr.meta.omero['channels']
            if hasattr(self.manager, 'pyr') and self.manager.pyr is not None
            else self.manager.channels
        )
        
        for idx, channel in enumerate(mchannels):
            try:
                default_channels[idx].update(channel)
            except Exception as e:
                logger.error(f"Failed to update channel {idx} with {channel}: {e}")
    
    def _apply_intensity_windows(self):
        """Apply intensity limits and normalize color codes for all channels."""
        assert self.channel_intensity_limits in ('from_dtype', 'from_array', 'auto'), \
            f"Channel intensity limits must be 'from_dtype', 'from_array' or 'auto'"
        
        from_array = self.channel_intensity_limits == 'from_array'
        from_none = self.channel_intensity_limits == 'auto'
        
        # Compute intensity values
        start_intensities, end_intensities = \
            self.manager.compute_intensity_limits(from_array=from_array)
        mins, maxes = self.manager.compute_intensity_extrema(dtype=self.manager.array.dtype)
        
        # Apply to each channel
        for idx in range(len(self.output)):
            channel = self.output[idx]
            self._normalize_color(channel)
            
            if not from_none:
                channel['window'] = {
                    'min': mins[idx],
                    'max': maxes[idx],
                    'start': start_intensities[idx],
                    'end': end_intensities[idx]
                }
    
    def _normalize_color(self, channel: dict):
        """Normalize hex color code to 6-character format.
        
        Args:
            channel: Channel metadata dictionary to modify in-place
        """
        color = channel['color']
        
        if color.startswith('#'):
            color = color[1:]
        
        if len(color) == 6:
            pass  # Already correct
        elif len(color) == 12:
            logger.warning("Color code parsed from 12-hex to 6-hex format")
            color = color[::2]
        else:
            logger.warning("Color code doesn't follow hex format, truncating to 6 characters")
            color = color[:6]
        
        channel['color'] = color
    
    def _apply_user_customizations(self):
        """Parse, validate, and apply user-provided customizations."""
        indices, labels, colors = self._parse_user_parameters()
    
    def _apply_user_customizations(self):
        """Parse, validate, and apply user-provided customizations."""
        # Parse indexed signature format for labels and colors
        label_dict = self._parse_indexed_string(
            self.kwargs.get('channel_labels', '')
        )
        color_dict = self._parse_indexed_string(
            self.kwargs.get('channel_colors', '')
        )
        
        # Early return if no customizations requested
        if not label_dict and not color_dict:
            return
        
        self._apply_customizations(label_dict, color_dict)
    
    def _parse_indexed_string(self, formatted_str: str) -> dict:
        """Parse indexed signature format into dictionary.
        
        Format: "idx1,value1;idx2,value2;idx3,value3"
        Example: "0,Red;1,Green;2,Blue"
        
        Args:
            formatted_str: String with signature format, or None/empty
            
        Returns:
            Dictionary mapping channel indices to values
            
        Raises:
            ValueError: If format is invalid
        """
        if not formatted_str or formatted_str is None:
            return {}
        
        result = {}
        try:
            for pair in formatted_str.split(';'):
                pair = pair.strip()
                if not pair:
                    continue
                
                parts = pair.split(',', 1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid pair '{pair}', expected 'index,value'")
                
                idx_str, value = parts
                try:
                    idx = int(idx_str.strip())
                except ValueError:
                    raise ValueError(f"Invalid index '{idx_str}', must be integer")
                
                result[idx] = value.strip()
        except Exception as e:
            raise ValueError(f"Failed to parse format string '{formatted_str}': {e}")
        
        return result
    
    def _apply_customizations(self, label_dict: dict, color_dict: dict):
        """Apply label and color customizations to specified channels.
        
        Args:
            label_dict: Dict mapping channel indices to labels
            color_dict: Dict mapping channel indices to colors
        """
        cm = ChannelMap()
        
        # Collect all indices to update
        all_indices = set(label_dict.keys()) | set(color_dict.keys())
        
        for channel_idx in sorted(all_indices):
            # Validate index is in range
            if channel_idx >= self.channel_count or channel_idx < 0:
                logger.warning(
                    f"Channel {channel_idx} out of range [0:{self.channel_count-1}] "
                    f"for {self.manager.series_path}, skipping"
                )
                continue
            
            channel = self.output[channel_idx]
            
            # Apply label if provided
            if channel_idx in label_dict:
                label = label_dict[channel_idx]
                if label not in ('', 'auto'):
                    channel['label'] = label
            
            # Apply color if provided (convert color name to hex if needed)
            if channel_idx in color_dict:
                colorname = color_dict[channel_idx]
                if colorname not in ('', 'auto'):
                    # Try to convert color name to hex
                    hex_color = cm[colorname]
                    channel['color'] = hex_color if hex_color is not None else colorname


def parse_channels(manager, **kwargs):
    """Parse and customize channel metadata.
    
    Handles channel initialization, intensity window computation, and user
    customizations (labels, colors).
    
    Args:
        manager: ArrayManager instance with array and metadata
        **kwargs: Customization parameters:
            - dtype: Data type (default: from manager)
            - channel_labels: Labels in format "idx1,label1;idx2,label2;..." (default: '')
            - channel_colors: Colors in format "idx1,color1;idx2,color2;..." (default: '')
            - channel_intensity_limits: 'from_dtype', 'from_array', or 'auto'
    
    Returns:
        List of JSON-safe channel metadata dictionaries
        
    Examples:
        >>> parse_channels(manager, 
        ...     channel_labels="0,Red;1,Green;2,Blue",
        ...     channel_colors="0,FF0000;1,00FF00;2,0000FF")
    """
    return ChannelParser(manager, **kwargs).parse()

