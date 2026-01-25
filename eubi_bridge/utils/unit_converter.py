"""Unit of measurement conversion utilities."""

from typing import Optional


def abbreviate_units(measure: Optional[str]) -> Optional[str]:
    """Abbreviate a unit of measurement.
    
    Converts human-readable units (e.g., "millimeter") to abbreviations (e.g., "mm").
    
    Args:
        measure: Human-readable unit of measurement
    
    Returns:
        Abbreviated form, or the input if already abbreviated
    
    Examples:
        >>> abbreviate_units("millimeter")
        'mm'
        >>> abbreviate_units("micrometer")
        'µm'
        >>> abbreviate_units("second")
        's'
    """
    if measure is None:
        return None

    abbreviations = {
        # Length measurements
        "millimeter": "mm",
        "centimeter": "cm",
        "decimeter": "dm",
        "meter": "m",
        "decameter": "dam",
        "hectometer": "hm",
        "kilometer": "km",
        "micrometer": "µm",
        "nanometer": "nm",
        "picometer": "pm",
        # Time measurements
        "second": "s",
        "millisecond": "ms",
        "microsecond": "µs",
        "nanosecond": "ns",
        "minute": "min",
        "hour": "h"
    }

    # Return the input if it's already an abbreviation
    if measure.lower() in abbreviations.values():
        return measure.lower()

    return abbreviations.get(measure.lower(), "Unknown")


def expand_units(measure: Optional[str]) -> Optional[str]:
    """Expand a unit abbreviation to full name.
    
    Converts abbreviations (e.g., "mm") to human-readable form (e.g., "millimeter").
    
    Args:
        measure: Unit abbreviation
    
    Returns:
        Full unit name, or the input if not recognized
    
    Examples:
        >>> expand_units("mm")
        'millimeter'
        >>> expand_units("µm")
        'micrometer'
        >>> expand_units("s")
        'second'
    """
    if measure is None:
        return None

    # Reverse mapping from abbreviations to full names
    expansions = {
        # Length measurements
        "mm": "millimeter",
        "cm": "centimeter",
        "dm": "decimeter",
        "m": "meter",
        "dam": "decameter",
        "hm": "hectometer",
        "km": "kilometer",
        "µm": "micrometer",
        "nm": "nanometer",
        "pm": "picometer",
        # Time measurements
        "s": "second",
        "ms": "millisecond",
        "µs": "microsecond",
        "ns": "nanosecond",
        "min": "minute",
        "h": "hour"
    }

    return expansions.get(measure.lower(), measure)
