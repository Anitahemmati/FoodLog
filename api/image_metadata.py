"""
Helpers for extracting EXIF metadata from uploaded images.

The module focuses on the subset of tags that may be informative for
food-analysis use cases (distance, orientation, camera parameters, GPS etc.).
All values are converted into JSON-serialisable primitives so the payload can
be persisted in DynamoDB without additional processing.
"""

from __future__ import annotations

import io
import logging
from fractions import Fraction
from typing import Any, Dict, Optional

from PIL import ExifTags, Image

logger = logging.getLogger(__name__)

# Build reverse lookups for EXIF GPS tag names.
GPS_TAGS = {key: value for key, value in ExifTags.GPSTAGS.items()}

# Orientation values defined by the EXIF spec mapped to degrees.
ORIENTATION_TO_DEGREES = {
  1: 0,
  3: 180,
  6: 90,
  8: 270,
}


def _ratio_to_float(value: Any) -> Optional[float]:
  """
  Convert EXIF rational tuples/fractions to floats.

  EXIF stores many numeric values as ``(numerator, denominator)`` tuples or
  ``fractions.Fraction`` instances. This helper normalises them to floats so
  they can be serialised into JSON. Non-numeric inputs return ``None``.
  """
  if isinstance(value, Fraction):
    try:
      return float(value)
    except (TypeError, ZeroDivisionError):
      return None

  if isinstance(value, tuple):
    if len(value) == 2 and value[1]:
      try:
        return float(value[0]) / float(value[1])
      except (TypeError, ZeroDivisionError):
        return None

    # GPS coordinates arrive as tuple of 3 rationals.
    if len(value) == 3:
      parts = [_ratio_to_float(part) for part in value]
      if all(part is not None for part in parts):
        degrees, minutes, seconds = parts  # type: ignore[assignment]
        return degrees + (minutes / 60.0) + (seconds / 3600.0)
    return None

  if isinstance(value, (int, float)):
    return float(value)

  return None


def _parse_gps(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
  """Return structured GPS metadata when available."""
  gps_raw = data.get("GPSInfo")
  if not gps_raw:
    return None

  gps_parsed = {}
  for key, value in gps_raw.items():
    tag = GPS_TAGS.get(key, str(key))
    gps_parsed[tag] = value

  latitude = None
  longitude = None

  lat_values = gps_parsed.get("GPSLatitude")
  lat_ref = gps_parsed.get("GPSLatitudeRef")
  lon_values = gps_parsed.get("GPSLongitude")
  lon_ref = gps_parsed.get("GPSLongitudeRef")

  if lat_values and lat_ref:
    lat_deg = _ratio_to_float(lat_values)
    if lat_deg is not None:
      latitude = lat_deg if lat_ref in {"N", "n"} else -lat_deg

  if lon_values and lon_ref:
    lon_deg = _ratio_to_float(lon_values)
    if lon_deg is not None:
      longitude = lon_deg if lon_ref in {"E", "e"} else -lon_deg

  result = {
    "latitude": latitude,
    "longitude": longitude,
    "altitude": _ratio_to_float(gps_parsed.get("GPSAltitude")),
    "timestamp": gps_parsed.get("GPSTimeStamp"),
  }

  # Drop keys with ``None`` values to keep payload tidy.
  return {key: value for key, value in result.items() if value is not None} or None


def extract_metadata(image_bytes: bytes) -> Dict[str, Any]:
  """
  Extract EXIF metadata from raw image bytes.

  Only a curated subset of tags are returned so the resulting structure stays
  concise and meaningful for API consumers.
  """
  metadata: Dict[str, Any] = {}

  try:
    with Image.open(io.BytesIO(image_bytes)) as image:
      exif_data = image.getexif()
      if not exif_data:
        return metadata

      for tag_id, value in exif_data.items():
        tag = ExifTags.TAGS.get(tag_id, str(tag_id))
        metadata[tag] = value
  except Exception as exc:
    logger.debug("Failed to parse image metadata: %s", exc)
    return {}

  result: Dict[str, Any] = {}

  # Camera and lens data.
  for key in ("Make", "Model", "LensModel", "LensSpecification"):
    if key in metadata:
      result[key[0].lower() + key[1:]] = metadata[key]

  # Exposure settings.
  focal_length = _ratio_to_float(metadata.get("FocalLength"))
  if focal_length is not None:
    result["focalLengthMm"] = round(focal_length, 4)

  f_number = _ratio_to_float(metadata.get("FNumber"))
  if f_number is not None:
    result["aperture"] = round(f_number, 4)

  exposure_time = metadata.get("ExposureTime")
  if exposure_time is not None:
    # Exposure time is often a (num, den) tuple.
    exposure_float = _ratio_to_float(exposure_time)
    if exposure_float is not None:
      result["exposureSeconds"] = round(exposure_float, 6)
    else:
      result["exposure"] = str(exposure_time)

  iso_value = metadata.get("ISOSpeedRatings") or metadata.get("PhotographicSensitivity")
  if isinstance(iso_value, (int, float)):
    result["iso"] = int(iso_value)

  subject_distance = _ratio_to_float(metadata.get("SubjectDistance"))
  if subject_distance is not None:
    result["subjectDistanceMeters"] = round(subject_distance, 4)

  orientation = metadata.get("Orientation")
  if isinstance(orientation, int):
    result["sceneRotationDegrees"] = ORIENTATION_TO_DEGREES.get(orientation, 0)

  capture_time = metadata.get("DateTimeOriginal") or metadata.get("DateTime")
  if capture_time:
    result["capturedAt"] = str(capture_time)

  gps_info = _parse_gps(metadata)
  if gps_info:
    result["gps"] = gps_info

  return result


__all__ = ["extract_metadata"]

