"""
OME-NGFF metadata generation utilities.
Handles creation of compliant NGFF metadata for multiscale images.
"""

import json
import time
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np


class NGFFMetadataBuilder:
    """Builder for OME-NGFF compliant metadata."""
    
    def __init__(self, version: str = "0.4"):
        self.version = version
        self.supported_versions = ["0.4", "0.5"]
        
        if version not in self.supported_versions:
            raise ValueError(f"Unsupported NGFF version: {version}. Supported: {self.supported_versions}")
    
    def build_multiscales_metadata(self, 
                                 name: str,
                                 axes: str,
                                 pyramid_levels: List[Dict[str, Any]],
                                 dtype: np.dtype,
                                 pixel_sizes: Optional[Dict[str, float]] = None,
                                 pixel_units: Optional[Dict[str, str]] = None,
                                 channel_names: Optional[List[str]] = None,
                                 time_points: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Build complete multiscales metadata for OME-NGFF.
        
        Args:
            name: Name of the dataset
            axes: Axis order string (e.g., "tczyx")
            pyramid_levels: List of pyramid level information
            dtype: Data type of the arrays
            pixel_sizes: Physical pixel sizes for spatial axes
            pixel_units: Physical pixel units for spatial axes
            channel_names: Names for channels (if C axis present)
            time_points: Time points (if T axis present)
        
        Returns:
            Complete NGFF metadata dictionary
        """
        # Build axes metadata
        axes_metadata = self._build_axes_metadata(axes,
                                                  pixel_units,
                                                  pixel_sizes)
        
        # Build datasets metadata for all pyramid levels
        datasets = self._build_datasets_metadata(pyramid_levels, axes, pixel_sizes)
        
        # Build coordinate transformations
        self._add_coordinate_transformations(datasets,
                                             pyramid_levels,
                                             axes,
                                             pixel_sizes,
                                             pixel_units
                                             )
        
        # Main multiscales entry
        multiscale_entry = {
            "version": self.version,
            "name": name,
            "datasets": datasets,
            "axes": axes_metadata,
            "type": "image"
        }
        
        # Add metadata for specific axes
        metadata = {}
        
        # Channel metadata
        if 'c' in axes and channel_names:
            metadata["channels"] = self._build_channel_metadata(channel_names)
        
        # Time metadata
        if 't' in axes and time_points:
            metadata["time"] = self._build_time_metadata(time_points)
        
        # Add metadata if any exists
        if metadata:
            multiscale_entry["metadata"] = metadata
        
        # Complete NGFF structure
        ngff_metadata = {
            "multiscales": [multiscale_entry]
        }
        
        # Add NGFF version at root level for newer versions
        if self.version == "0.5":
            ngff_metadata["ngff"] = {"version": self.version}
        
        return ngff_metadata
    
    def _build_axes_metadata(self,
                             axes: str,
                             pixel_units: Optional[Dict[str, str]] = None,
                             pixel_sizes: Optional[Dict[str, float]] = None
                             ) -> List[Dict[str, Any]]:
        """Build axes metadata array."""
        axes_list = []
        
        for axis in axes:
            axis_entry = self._get_axis_metadata(axis)
            
            # Add physical units for spatial axes
            if pixel_sizes and axis in pixel_sizes:
                # Physical size is handled in coordinate transformations
                # Here axis order and units are handled
                axis_entry["unit"] = pixel_units.get(axis, "micrometer")

            axes_list.append(axis_entry)
        
        return axes_list
    
    def _get_axis_metadata(self, axis: str) -> Dict[str, Any]:
        """Get metadata for a single axis."""
        axis_map = {
            't': {"name": "t", "type": "time", "unit": "second"},
            'c': {"name": "c", "type": "channel"},
            'z': {"name": "z", "type": "space", "unit": "micrometer"},
            'y': {"name": "y", "type": "space", "unit": "micrometer"},
            'x': {"name": "x", "type": "space", "unit": "micrometer"}
        }
        
        return axis_map.get(axis, {"name": axis, "type": "unknown"})
    
    def _build_datasets_metadata(self, pyramid_levels: List[Dict[str, Any]], 
                                axes: str, pixel_sizes: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """Build datasets array for all pyramid levels."""
        datasets = []
        
        for level_info in pyramid_levels:
            level = level_info["level"]
            
            dataset_entry = {
                "path": str(level)
            }
            
            datasets.append(dataset_entry)
        
        return datasets
    
    def _add_coordinate_transformations(self,
                                        datasets: List[Dict[str, Any]],
                                        pyramid_levels: List[Dict[str, Any]],
                                        axes: str,
                                        pixel_sizes: Optional[Dict[str, float]] = None,
                                        pixel_units: Optional[Dict[str, str]] = None
                                        ):
        """Add coordinate transformations to dataset entries."""
        for i, (dataset, level_info) in enumerate(zip(datasets, pyramid_levels)):
            level = level_info["level"]
            downsample_factors = level_info.get("downsample_factors", {})
            
            # Build scale transformation
            scale_values = []
            for axis in axes:
                # Get cumulative scale factor for this axis and level
                scale_factor = downsample_factors.get(axis, 1.0)
                
                # Apply physical pixel size if available
                if pixel_sizes and axis in pixel_sizes and axis in 'xyz':
                    physical_scale = pixel_sizes[axis] * scale_factor
                    scale_values.append(physical_scale)
                else:
                    # Use dimensionless scale factor
                    scale_values.append(float(scale_factor))
            
            # Create coordinate transformation
            coordinate_transform = {
                "type": "scale",
                "scale": scale_values
            }
            
            dataset["coordinateTransformations"] = [coordinate_transform]
    
    def _build_channel_metadata(self, channel_names: List[str]) -> List[Dict[str, Any]]:
        """Build channel metadata."""
        channels = []
        
        for i, name in enumerate(channel_names):
            channel_entry = {
                "label": name,
                "active": True
            }
            
            # Add default visualization settings
            if i < len(self._get_default_colors()):
                channel_entry["color"] = self._get_default_colors()[i]
            
            channels.append(channel_entry)
        
        return channels
    
    def _get_default_colors(self) -> List[str]:
        """Get default color palette for channels."""
        return [
            "00FFFF",  # Cyan
            "FF00FF",  # Magenta  
            "FFFF00",  # Yellow
            "FF0000",  # Red
            "00FF00",  # Green
            "0000FF",  # Blue
            "FF8000",  # Orange
            "8000FF",  # Purple
        ]
    
    def _build_time_metadata(self, time_points: List[float]) -> Dict[str, Any]:
        """Build time metadata."""
        return {
            "timepoints": time_points,
            "unit": "second"
        }
    
    def add_acquisition_metadata(self, metadata: Dict[str, Any], 
                               acquisition_info: Dict[str, Any]) -> Dict[str, Any]:
        """Add acquisition-specific metadata."""
        updated_metadata = metadata.copy()
        
        if "multiscales" not in updated_metadata:
            updated_metadata["multiscales"] = [{}]
        
        multiscale = updated_metadata["multiscales"][0]
        
        if "metadata" not in multiscale:
            multiscale["metadata"] = {}
        
        # Add acquisition metadata
        acquisition_metadata = {
            "acquisition": {
                "instrument": acquisition_info.get("instrument", "Unknown"),
                "objective": acquisition_info.get("objective", {}),
                "detector": acquisition_info.get("detector", {}),
                "timestamp": acquisition_info.get("timestamp", time.time())
            }
        }
        
        multiscale["metadata"].update(acquisition_metadata)
        
        return updated_metadata
    
    def add_processing_metadata(self, metadata: Dict[str, Any], 
                              processing_info: Dict[str, Any]) -> Dict[str, Any]:
        """Add processing history metadata."""
        updated_metadata = metadata.copy()
        
        if "multiscales" not in updated_metadata:
            updated_metadata["multiscales"] = [{}]
        
        multiscale = updated_metadata["multiscales"][0]
        
        if "metadata" not in multiscale:
            multiscale["metadata"] = {}
        
        # Add processing metadata
        processing_metadata = {
            "processing": {
                "software": processing_info.get("software", "bigtiff-to-ngff-converter"),
                "version": processing_info.get("version", "1.0.0"),
                "timestamp": processing_info.get("timestamp", time.time()),
                "parameters": processing_info.get("parameters", {}),
                "source_file": processing_info.get("source_file"),
                "conversion_time": processing_info.get("conversion_time")
            }
        }
        
        multiscale["metadata"].update(processing_metadata)
        
        return updated_metadata
    
    def add_pixel_size_metadata(self, metadata: Dict[str, Any], 
                              pixel_sizes: Dict[str, float],
                              units: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Add detailed pixel size metadata."""
        updated_metadata = metadata.copy()
        
        if "multiscales" not in updated_metadata:
            updated_metadata["multiscales"] = [{}]
        
        multiscale = updated_metadata["multiscales"][0]
        
        if "metadata" not in multiscale:
            multiscale["metadata"] = {}
        
        # Build pixel size metadata
        pixel_size_metadata = {"pixel_sizes": {}}
        
        for axis, size in pixel_sizes.items():
            axis_metadata = {"value": size}
            
            if units and axis in units:
                axis_metadata["unit"] = units[axis]
            elif axis in 'xyz':
                axis_metadata["unit"] = "micrometer"
            elif axis == 't':
                axis_metadata["unit"] = "second"
            
            pixel_size_metadata["pixel_sizes"][axis] = axis_metadata
        
        multiscale["metadata"].update(pixel_size_metadata)
        
        return updated_metadata
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate NGFF metadata and return validation results."""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "version": self.version
        }
        
        # Check required top-level structure
        if "multiscales" not in metadata:
            validation_results["errors"].append("Missing required 'multiscales' field")
            validation_results["valid"] = False
            return validation_results
        
        multiscales = metadata["multiscales"]
        if not isinstance(multiscales, list) or len(multiscales) == 0:
            validation_results["errors"].append("'multiscales' must be a non-empty array")
            validation_results["valid"] = False
            return validation_results
        
        # Validate each multiscale entry
        for i, multiscale in enumerate(multiscales):
            prefix = f"multiscales[{i}]"
            
            # Check required fields
            required_fields = ["datasets", "axes"]
            for field in required_fields:
                if field not in multiscale:
                    validation_results["errors"].append(f"{prefix}: Missing required field '{field}'")
                    validation_results["valid"] = False
            
            # Validate version
            if "version" in multiscale:
                version = multiscale["version"]
                if version not in self.supported_versions:
                    validation_results["warnings"].append(
                        f"{prefix}: Version '{version}' may not be fully supported"
                    )
            
            # Validate datasets
            if "datasets" in multiscale:
                datasets = multiscale["datasets"]
                if not isinstance(datasets, list):
                    validation_results["errors"].append(f"{prefix}: 'datasets' must be an array")
                    validation_results["valid"] = False
                else:
                    for j, dataset in enumerate(datasets):
                        dataset_prefix = f"{prefix}.datasets[{j}]"
                        
                        if "path" not in dataset:
                            validation_results["errors"].append(
                                f"{dataset_prefix}: Missing required 'path' field"
                            )
                            validation_results["valid"] = False
                        
                        if "coordinateTransformations" not in dataset:
                            validation_results["warnings"].append(
                                f"{dataset_prefix}: Missing 'coordinateTransformations'"
                            )
            
            # Validate axes
            if "axes" in multiscale:
                axes = multiscale["axes"]
                if not isinstance(axes, list):
                    validation_results["errors"].append(f"{prefix}: 'axes' must be an array")
                    validation_results["valid"] = False
                else:
                    for j, axis in enumerate(axes):
                        axis_prefix = f"{prefix}.axes[{j}]"
                        
                        if not isinstance(axis, dict):
                            validation_results["errors"].append(
                                f"{axis_prefix}: Axis must be an object"
                            )
                            validation_results["valid"] = False
                            continue
                        
                        if "name" not in axis:
                            validation_results["errors"].append(
                                f"{axis_prefix}: Missing required 'name' field"
                            )
                            validation_results["valid"] = False
                        
                        if "type" not in axis:
                            validation_results["warnings"].append(
                                f"{axis_prefix}: Missing recommended 'type' field"
                            )
        
        return validation_results
    
    def convert_metadata_version(self, metadata: Dict[str, Any], 
                                target_version: str) -> Dict[str, Any]:
        """Convert metadata between NGFF versions."""
        if target_version not in self.supported_versions:
            raise ValueError(f"Unsupported target version: {target_version}")
        
        converted_metadata = metadata.copy()
        
        # Version-specific conversions
        if target_version == "0.5":
            # Add top-level version field for 0.5
            converted_metadata["ngff"] = {"version": target_version}
            
            # Update multiscale version fields
            if "multiscales" in converted_metadata:
                for multiscale in converted_metadata["multiscales"]:
                    multiscale["version"] = target_version
        
        elif target_version == "0.4":
            # Remove top-level version field for 0.4
            if "ngff" in converted_metadata:
                del converted_metadata["ngff"]
            
            # Update multiscale version fields
            if "multiscales" in converted_metadata:
                for multiscale in converted_metadata["multiscales"]:
                    multiscale["version"] = target_version
        
        return converted_metadata
    
    def extract_pixel_sizes_from_metadata(self, metadata: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Extract pixel sizes from existing NGFF metadata."""
        if "multiscales" not in metadata:
            return None
        
        multiscale = metadata["multiscales"][0]
        
        # Try to extract from coordinate transformations
        if "datasets" in multiscale and multiscale["datasets"]:
            dataset = multiscale["datasets"][0]  # Use first (highest resolution) dataset
            
            if "coordinateTransformations" in dataset:
                for transform in dataset["coordinateTransformations"]:
                    if transform.get("type") == "scale" and "scale" in transform:
                        scales = transform["scale"]
                        axes = multiscale.get("axes", [])
                        
                        pixel_sizes = {}
                        for i, axis_info in enumerate(axes):
                            if i < len(scales):
                                axis_name = axis_info.get("name", f"axis_{i}")
                                pixel_sizes[axis_name] = scales[i]
                        
                        return pixel_sizes
        
        # Try to extract from metadata section
        if "metadata" in multiscale and "pixel_sizes" in multiscale["metadata"]:
            pixel_size_metadata = multiscale["metadata"]["pixel_sizes"]
            pixel_sizes = {}
            
            for axis, size_info in pixel_size_metadata.items():
                if isinstance(size_info, dict) and "value" in size_info:
                    pixel_sizes[axis] = size_info["value"]
                else:
                    pixel_sizes[axis] = float(size_info)
            
            return pixel_sizes
        
        return None


def create_basic_ngff_metadata(name: str, shape: Tuple[int, ...], 
                              axes: str, dtype: np.dtype) -> Dict[str, Any]:
    """Create basic NGFF metadata for simple cases."""
    builder = NGFFMetadataBuilder()
    
    # Create single pyramid level
    pyramid_levels = [{
        "level": 0,
        "shape": shape,
        "downsample_factors": {axis: 1 for axis in axes}
    }]
    
    return builder.build_multiscales_metadata(
        name=name,
        axes=axes,
        pyramid_levels=pyramid_levels,
        dtype=dtype
    )


def update_ngff_metadata_with_stats(metadata: Dict[str, Any], 
                                   conversion_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Update NGFF metadata with conversion statistics."""
    builder = NGFFMetadataBuilder()
    
    processing_info = {
        "software": "High-Performance BigTIFF to OME-NGFF Converter",
        "version": "1.0.0",
        "timestamp": time.time(),
        "conversion_time": conversion_stats.get("duration", 0),
        "throughput_mb_s": conversion_stats.get("throughput_mb_s", 0),
        "compression_ratio": conversion_stats.get("compression_ratio", 1.0),
        "parameters": {
            "workers": conversion_stats.get("workers"),
            "compression": conversion_stats.get("compression"),
            "chunk_memory_mb": conversion_stats.get("chunk_memory_mb")
        }
    }
    
    return builder.add_processing_metadata(metadata, processing_info)


def validate_ngff_compliance(metadata: Dict[str, Any], 
                           version: str = "0.4") -> Dict[str, Any]:
    """Validate NGFF metadata compliance."""
    builder = NGFFMetadataBuilder(version=version)
    return builder.validate_metadata(metadata)
