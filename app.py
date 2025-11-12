"""
Flask backend for the Personal Food Log App.

This service accepts meal photos uploaded from the React frontend, stores them
in either local storage (for development) or Amazon S3 (for production),
enriches them with EXIF-derived metadata, calls the external ML service for
nutrition predictions (with a local fallback), persists the final result in
SQLite or DynamoDB, and exposes an API for retrieving historical entries.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from difflib import get_close_matches
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory, abort
from flask_cors import CORS
import requests
from werkzeug.security import check_password_hash, generate_password_hash

import jwt

from jwt import ExpiredSignatureError, InvalidTokenError
from api.image_metadata import extract_metadata
from api.ml_predict import predict_calories
from api.ml_service import MLServiceError, call_ml_service

LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = Path(os.environ.get("UPLOADS_DIR", str(BASE_DIR / "uploads"))).resolve()
DB_PATH = Path(os.environ.get("SQLITE_DB_PATH", str(BASE_DIR / "food_history.db"))).resolve()

load_dotenv(BASE_DIR.parent / ".env")

STORAGE_BACKEND = os.environ.get("STORAGE_BACKEND", "sqlite").strip().lower()
USE_AWS_BACKEND = STORAGE_BACKEND == "aws"

JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_MINUTES = int(os.environ.get("JWT_EXPIRATION_MINUTES", "60"))
DEFAULT_HF_SPACE_URL = "https://router.huggingface.co/hf-inference/models/nateraw/food"


def _safe_float(value: Optional[str], default: float) -> float:
  try:
    return float(value) if value is not None else default
  except (TypeError, ValueError):
    return default


def _safe_int(value: Optional[str], default: int) -> int:
  try:
    return int(value) if value is not None else default
  except (TypeError, ValueError):
    return default


def _normalise_confidence_value(value: float) -> float:
  """Return confidence in the 0-1 range even if expressed as a percentage."""
  if value > 1:
    return min(1.0, value / 100.0)
  if value < 0:
    return 0.0
  return value


def _parse_csv(value: Optional[str]) -> List[str]:
  """Return a list of comma-separated values with whitespace removed."""
  if not value:
    return []
  return [item.strip() for item in value.split(",") if item.strip()]


def _normalise_hf_url(url: Optional[str]) -> str:
  """Rewrite deprecated Hugging Face endpoints to the router host."""
  cleaned = (url or "").strip()
  if not cleaned:
    return ""

  parsed = urlparse(cleaned)
  netloc = parsed.netloc.lower()
  path = parsed.path or ""
  if "api-inference.huggingface.co" in netloc:
    if path.startswith("/models/") and not path.startswith("/hf-inference"):
      path = "/hf-inference" + path
    elif not path.startswith("/hf-inference"):
      path = "/hf-inference/models/nateraw/food"
    parsed = parsed._replace(scheme="https", netloc="router.huggingface.co", path=path)
    cleaned = parsed.geturl()
  return cleaned


HF_SPACE_URL = _normalise_hf_url(os.environ.get("HF_SPACE_URL") or os.environ.get("HF_FOOD_SPACE_URL") or DEFAULT_HF_SPACE_URL)
HF_CONFIDENCE_THRESHOLD = _normalise_confidence_value(_safe_float(os.environ.get("HF_CONFIDENCE_THRESHOLD"), 0.5))
HF_SPACE_TIMEOUT = _safe_int(os.environ.get("HF_SPACE_TIMEOUT"), 45)
def _sanitise_status_code(value: int, default: int = 200) -> int:
  try:
    if 100 <= value <= 599:
      return value
  except TypeError:
    pass
  return default

HF_API_TOKEN = (
  os.environ.get("HF_API_TOKEN")
  or os.environ.get("HUGGING_FACE_TOKEN")
  or os.environ.get("HUGGINGFACE_TOKEN")
  or os.environ.get("HF_TOKEN")
  or ""
).strip()
HF_REJECTION_STATUS_CODE = _sanitise_status_code(_safe_int(os.environ.get("HF_REJECTION_STATUS_CODE"), 200))

FDC_API_KEY = (
  os.environ.get("FDC_API_KEY")
  or os.environ.get("USDA_API_KEY")
  or ""
).strip()
FDC_ENDPOINT = "https://api.nal.usda.gov/fdc/v1/foods/search"
FDC_PAGE_SIZE = max(1, min(25, _safe_int(os.environ.get("FDC_PAGE_SIZE"), 3)))
FDC_TIMEOUT = _safe_int(os.environ.get("FDC_TIMEOUT"), 8)
FDC_DATA_TYPES = _parse_csv(os.environ.get("FDC_DATA_TYPES")) or [
  "Survey (FNDDS)",
  "SR Legacy",
  "Branded",
]
FDC_BRAND_OWNER = (os.environ.get("FDC_BRAND_OWNER") or "").strip()


def _normalise_label_key(value: Any) -> str:
  """Return a normalised lookup key for food labels."""
  if value is None:
    return ""
  cleaned = re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower())
  return " ".join(cleaned.split())


def _resolve_ingredient_override(label: Any) -> List[str] | None:
  """Return curated ingredients for labels that approximately match known meals."""
  key = _normalise_label_key(label)
  if not key:
    return None

  direct = HF_INGREDIENT_OVERRIDES.get(key)
  if direct:
    return direct

  ordered_keys = sorted(HF_INGREDIENT_OVERRIDES.keys(), key=len, reverse=True)
  for candidate_key in ordered_keys:
    ingredients = HF_INGREDIENT_OVERRIDES[candidate_key]
    if candidate_key in key or key in candidate_key:
      return ingredients

  close_matches = get_close_matches(
    key,
    list(HF_INGREDIENT_OVERRIDES.keys()),
    n=1,
    cutoff=0.82,
  )
  if close_matches:
    return HF_INGREDIENT_OVERRIDES.get(close_matches[0])

  return None


HF_INGREDIENT_OVERRIDES: Dict[str, List[str]] = {
  # Core templates (kept in sync with synthetic fallback).
  "grilled chicken salad": [
    "Grilled Chicken Breast",
    "Mixed Greens",
    "Cherry Tomatoes",
    "Cucumber",
    "Balsamic Vinaigrette",
  ],
  "veggie power bowl": [
    "Quinoa",
    "Roasted Sweet Potato",
    "Black Beans",
    "Avocado",
    "Lime Dressing",
  ],
  "salmon sushi roll": [
    "Sushi Rice",
    "Fresh Salmon",
    "Nori",
    "Cucumber",
    "Soy Sauce",
  ],
  "mediterranean wrap": [
    "Whole Wheat Tortilla",
    "Hummus",
    "Feta Cheese",
    "Kalamata Olives",
    "Spinach",
  ],
  "spaghetti bolognese": [
    "Spaghetti",
    "Tomato Sauce",
    "Ground Beef",
    "Parmesan Cheese",
    "Basil",
  ],
  # Additional common meals.
  "pepperoni pizza": [
    "Pizza Dough",
    "Tomato Sauce",
    "Mozzarella Cheese",
    "Pepperoni",
    "Olive Oil",
  ],
  "hamburger": [
    "Beef Patty",
    "Burger Bun",
    "Cheddar Cheese",
    "Lettuce",
    "Tomato",
  ],
  "cheeseburger": [
    "Beef Patty",
    "Burger Bun",
    "Cheddar Cheese",
    "Lettuce",
    "Tomato",
  ],
  "margherita pizza": [
    "Pizza Dough",
    "Tomato Sauce",
    "Fresh Mozzarella",
    "Basil Leaves",
    "Olive Oil",
  ],
  "caesar salad": [
    "Romaine Lettuce",
    "Grilled Chicken",
    "Parmesan Shavings",
    "Croutons",
    "Caesar Dressing",
  ],
  "chicken burrito": [
    "Flour Tortilla",
    "Grilled Chicken",
    "Rice",
    "Black Beans",
    "Salsa",
  ],
  "falafel wrap": [
    "Pita Bread",
    "Falafel",
    "Lettuce",
    "Tomato",
    "Tahini Sauce",
  ],
  "avocado toast": [
    "Sourdough Bread",
    "Mashed Avocado",
    "Cherry Tomatoes",
    "Olive Oil",
    "Sea Salt",
  ],
  "greek yogurt parfait": [
    "Greek Yogurt",
    "Granola",
    "Honey",
    "Blueberries",
    "Almonds",
  ],
  "fruit smoothie": [
    "Banana",
    "Mixed Berries",
    "Greek Yogurt",
    "Almond Milk",
    "Chia Seeds",
  ],
  "pizza": [
    "Pizza Dough",
    "Tomato Sauce",
    "Mozzarella Cheese",
    "Olive Oil",
    "Fresh Basil",
  ],
  "burger": [
    "Beef Patty",
    "Burger Bun",
    "Cheddar Cheese",
    "Lettuce",
    "Tomato",
  ],
  "sandwich": [
    "Whole Wheat Bread",
    "Turkey Slices",
    "Lettuce",
    "Tomato",
    "Mayonnaise",
  ],
  "wrap": [
    "Flour Tortilla",
    "Grilled Chicken",
    "Mixed Greens",
    "Tomato",
    "Yogurt Sauce",
  ],
  "salad": [
    "Mixed Greens",
    "Cherry Tomatoes",
    "Cucumber",
    "Red Onion",
    "Vinaigrette",
  ],
  "soup": [
    "Vegetable Broth",
    "Carrots",
    "Celery",
    "Onion",
    "Parsley",
  ],
  "pasta": [
    "Spaghetti",
    "Marinara Sauce",
    "Parmesan",
    "Olive Oil",
    "Garlic",
  ],
  "smoothie": [
    "Banana",
    "Mixed Berries",
    "Greek Yogurt",
    "Spinach",
    "Almond Milk",
  ],
  "omelette": [
    "Eggs",
    "Cheddar Cheese",
    "Spinach",
    "Mushrooms",
    "Bell Peppers",
  ],
  "pancakes": [
    "Flour",
    "Milk",
    "Eggs",
    "Maple Syrup",
    "Butter",
  ],
}


def _build_weight_map(pairs: List[Tuple[str, float]]) -> Dict[str, float]:
  """Return a normalised ingredient-to-grams map for USDA weighting."""
  weight_map: Dict[str, float] = {}
  for name, grams in pairs:
    key = _normalise_label_key(name)
    if not key:
      continue
    weight_map[key] = float(grams)
  return weight_map


HF_INGREDIENT_WEIGHT_OVERRIDES: Dict[str, Dict[str, float]] = {
  "grilled chicken salad": _build_weight_map(
    [
      ("Grilled Chicken Breast", 120.0),
      ("Mixed Greens", 80.0),
      ("Cherry Tomatoes", 50.0),
      ("Cucumber", 40.0),
      ("Balsamic Vinaigrette", 30.0),
    ]
  ),
  "pepperoni pizza": _build_weight_map(
    [
      ("Pizza Dough", 120.0),
      ("Tomato Sauce", 60.0),
      ("Mozzarella Cheese", 80.0),
      ("Pepperoni", 35.0),
      ("Olive Oil", 10.0),
    ]
  ),
  "margherita pizza": _build_weight_map(
    [
      ("Pizza Dough", 115.0),
      ("Tomato Sauce", 55.0),
      ("Fresh Mozzarella", 85.0),
      ("Basil Leaves", 5.0),
      ("Olive Oil", 8.0),
    ]
  ),
  "burger": _build_weight_map(
    [
      ("Beef Patty", 110.0),
      ("Burger Bun", 70.0),
      ("Cheddar Cheese", 25.0),
      ("Lettuce", 20.0),
      ("Tomato", 25.0),
    ]
  ),
  "hamburger": _build_weight_map(
    [
      ("Beef Patty", 110.0),
      ("Burger Bun", 70.0),
      ("Cheddar Cheese", 25.0),
      ("Lettuce", 20.0),
      ("Tomato", 25.0),
    ]
  ),
  "cheeseburger": _build_weight_map(
    [
      ("Beef Patty", 120.0),
      ("Burger Bun", 70.0),
      ("Cheddar Cheese", 35.0),
      ("Lettuce", 15.0),
      ("Tomato", 20.0),
    ]
  ),
  "sandwich": _build_weight_map(
    [
      ("Whole Wheat Bread", 60.0),
      ("Turkey Slices", 75.0),
      ("Lettuce", 20.0),
      ("Tomato", 25.0),
      ("Mayonnaise", 15.0),
    ]
  ),
  "salad": _build_weight_map(
    [
      ("Mixed Greens", 90.0),
      ("Cherry Tomatoes", 45.0),
      ("Cucumber", 40.0),
      ("Red Onion", 15.0),
      ("Vinaigrette", 25.0),
    ]
  ),
  "pasta": _build_weight_map(
    [
      ("Spaghetti", 140.0),
      ("Marinara Sauce", 90.0),
      ("Parmesan", 20.0),
      ("Olive Oil", 10.0),
      ("Garlic", 5.0),
    ]
  ),
  "falafel wrap": _build_weight_map(
    [
      ("Pita Bread", 80.0),
      ("Falafel", 70.0),
      ("Lettuce", 20.0),
      ("Tomato", 25.0),
      ("Tahini Sauce", 25.0),
    ]
  ),
}


def _extract_usda_macros(food_entry: Dict[str, Any]) -> Dict[str, Any] | None:
  """Return calories/macros parsed from a USDA FoodData Central entry."""
  nutrients = food_entry.get("foodNutrients") or []
  values: Dict[str, float] = {}
  for nutrient in nutrients:
    number = str(nutrient.get("nutrientNumber") or "").strip()
    if not number:
      continue
    value = nutrient.get("value")
    try:
      numeric_value = float(value)
    except (TypeError, ValueError):
      continue
    values[number] = numeric_value
    name = (nutrient.get("nutrientName") or "").lower()
    if number == "" and "energy" in name and "kcal" in name:
      values.setdefault("208", numeric_value)

  calories = values.get("208")
  if calories is None:
    return None

  def _round(value: float | None) -> int:
    if value is None:
      return 0
    return int(round(value))

  macros = {
    "calories": _round(calories),
    "proteins": _round(values.get("203")),
    "fats": _round(values.get("204")),
    "carbohydrates": _round(values.get("205")),
  }
  return macros


def _fetch_usda_nutrition(food_label: str, *, weight_grams: float | None = None) -> Dict[str, Any] | None:
  """Query FoodData Central for nutrition facts that match the supplied label."""
  if not FDC_API_KEY:
    return None
  normalised_label = (food_label or "").strip()
  if not normalised_label:
    return None

  payload: Dict[str, Any] = {
    "query": normalised_label,
    "pageSize": FDC_PAGE_SIZE,
    "requireAllWords": False,
    "sortBy": "score",
    "sortOrder": "desc",
  }
  if FDC_DATA_TYPES:
    payload["dataType"] = FDC_DATA_TYPES
  if FDC_BRAND_OWNER:
    payload["brandOwner"] = FDC_BRAND_OWNER

  try:
    response = requests.post(
      FDC_ENDPOINT,
      params={"api_key": FDC_API_KEY},
      json=payload,
      timeout=FDC_TIMEOUT,
    )
  except requests.RequestException as exc:
    LOGGER.info("USDA nutrition lookup failed: %s", exc)
    return None

  if not response.ok:
    LOGGER.info("USDA nutrition lookup error %s: %s", response.status_code, response.text[:120])
    return None

  try:
    content = response.json()
  except ValueError:
    return None

  foods = content.get("foods") or []
  for food in foods:
    macros = _extract_usda_macros(food)
    if not macros:
      continue
    portion_weight = None
    if weight_grams:
      scale = max(weight_grams, 0.0) / 100.0
      macros = {
        "calories": int(round(macros.get("calories", 0) * scale)),
        "proteins": int(round(macros.get("proteins", 0) * scale)),
        "fats": int(round(macros.get("fats", 0) * scale)),
        "carbohydrates": int(round(macros.get("carbohydrates", 0) * scale)),
      }
      portion_weight = weight_grams
    else:
      serving_size = food.get("servingSize")
      serving_unit = (food.get("servingSizeUnit") or "").strip().lower()
      try:
        serving_value = float(serving_size)
      except (TypeError, ValueError):
        serving_value = None
      if serving_value is not None:
        if serving_unit in {"g", "gram", "grams"}:
          portion_weight = serving_value
        elif serving_unit in {"oz", "ounce", "ounces"}:
          portion_weight = serving_value * 28.3495
    metadata = {
      "fdc_id": food.get("fdcId"),
      "description": food.get("description"),
      "data_type": food.get("dataType"),
      "brand_owner": food.get("brandOwner"),
      "published": food.get("publicationDate"),
      "serving_size": food.get("servingSize"),
      "serving_unit": food.get("servingSizeUnit"),
      "food_category": food.get("foodCategory"),
      "score": food.get("score"),
      "query": normalised_label,
      "serving_size": food.get("servingSize"),
      "serving_size_unit": food.get("servingSizeUnit"),
    }
    return {
      "calories": macros["calories"],
      "nutrition_facts": macros,
      "metadata": metadata,
      "weight_grams": portion_weight,
    }

  return None


def _aggregate_usda_from_ingredients(food_name: str, ingredients: List[str]) -> Dict[str, Any] | None:
  """Use per-ingredient USDA lookups when we know approximate serving weights."""
  if not FDC_API_KEY:
    return None
  meal_weights = HF_INGREDIENT_WEIGHT_OVERRIDES.get(_normalise_label_key(food_name))
  if not meal_weights:
    return None

  totals = {
    "calories": 0,
    "proteins": 0,
    "fats": 0,
    "carbohydrates": 0,
  }
  total_weight = 0.0
  components: List[Dict[str, Any]] = []
  for ingredient in ingredients:
    weight = meal_weights.get(_normalise_label_key(ingredient))
    if not weight:
      continue
    payload = _fetch_usda_nutrition(ingredient, weight_grams=weight)
    if not payload:
      continue
    macros = payload.get("nutrition_facts") or {}
    for key in totals:
      totals[key] += int(macros.get(key, 0))
    total_weight += weight
    component_meta = payload.get("metadata") or {}
    components.append(
      {
        "ingredient": ingredient,
        "weight_grams": weight,
        "calories": macros.get("calories"),
        "proteins": macros.get("proteins"),
        "fats": macros.get("fats"),
        "carbohydrates": macros.get("carbohydrates"),
        "fdc_id": component_meta.get("fdc_id"),
      }
    )

  if not components:
    return None

  totals = {key: int(round(value)) for key, value in totals.items()}
  metadata = {
    "strategy": "ingredient_synthesis",
    "components": components,
    "food_label": food_name,
  }
  return {
    "calories": totals["calories"],
    "nutrition_facts": totals,
    "metadata": metadata,
    "weight_grams": total_weight if total_weight else None,
  }


class HuggingFaceSpaceError(RuntimeError):
  """Raised when Hugging Face classification cannot be completed."""


def _extract_hf_label_score(payload: Any) -> Tuple[str, float]:
  """Return the highest-confidence label emitted by the HF space."""
  candidates: List[Tuple[str, float]] = []

  def _register(label: Any, score: Any) -> None:
    if not label or score is None:
      return
    try:
      numeric_score = _normalise_confidence_value(float(score))
    except (TypeError, ValueError):
      return
    candidates.append((str(label), numeric_score))

  def _walk_list(items: Any) -> None:
    if not isinstance(items, list):
      return
    for item in items:
      if isinstance(item, dict):
        label = item.get("label") or item.get("food") or item.get("class")
        score = item.get("confidence") or item.get("score") or item.get("similarity")
        if label and score is not None:
          _register(label, score)
          continue
        _walk_list(item.get("confidences"))
      elif isinstance(item, (list, tuple)):
        if len(item) >= 2:
          _register(item[0], item[1])
      elif isinstance(item, str):
        _register(item, 1.0)

  if isinstance(payload, list):
    _walk_list(payload)
  else:
    data_section = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data_section, dict):
      confidences = data_section.get("confidences")
      if confidences:
        _walk_list(confidences if isinstance(confidences, list) else [confidences])
      else:
        _register(
          data_section.get("label") or data_section.get("food"),
          data_section.get("confidence") or data_section.get("score"),
        )
    else:
      _walk_list(data_section if isinstance(data_section, list) else payload if isinstance(payload, list) else [])

  if not candidates:
    raise HuggingFaceSpaceError("Hugging Face space response did not include confidences.")
  return max(candidates, key=lambda candidate: candidate[1])


def _call_hf_food_space(image_bytes: bytes) -> Tuple[str, float, Dict[str, Any]]:
  """Invoke the Hugging Face space with the supplied image bytes."""
  if not HF_SPACE_URL:
    raise HuggingFaceSpaceError("Hugging Face space URL is not configured.")

  parsed_url = urlparse(HF_SPACE_URL)
  netloc = parsed_url.netloc.lower()
  path = parsed_url.path or ""
  is_inference_endpoint = (
    "api-inference.huggingface.co" in netloc
    or "router.huggingface.co" in netloc
    or path.startswith("/hf-inference")
  )

  headers = {}
  if HF_API_TOKEN:
    headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

  try:
    if is_inference_endpoint:
      headers.setdefault("Accept", "application/json")
      headers.setdefault("Content-Type", "application/octet-stream")
      response = requests.post(
        HF_SPACE_URL,
        headers=headers,
        data=image_bytes,
        timeout=HF_SPACE_TIMEOUT,
      )
    else:
      encoded = base64.b64encode(image_bytes).decode("utf-8")
      payload = {"data": [f"data:image/jpeg;base64,{encoded}"]}
      headers["Content-Type"] = "application/json"
      response = requests.post(
        HF_SPACE_URL,
        headers=headers,
        json=payload,
        timeout=HF_SPACE_TIMEOUT,
      )
  except requests.RequestException as exc:
    raise HuggingFaceSpaceError(f"Hugging Face request failed: {exc}") from exc

  if not response.ok:
    error_body = response.text[:200] if response.text else response.reason
    raise HuggingFaceSpaceError(f"Hugging Face space returned {response.status_code}: {error_body}")

  try:
    parsed = response.json()
  except ValueError as exc:
    raise HuggingFaceSpaceError("Hugging Face space response was not JSON.") from exc

  if isinstance(parsed, dict) and parsed.get("error"):
    raise HuggingFaceSpaceError(f"Hugging Face space error: {parsed['error']}")

  label, confidence = _extract_hf_label_score(parsed)
  return label, confidence, parsed if isinstance(parsed, dict) else {"data": parsed}


def _build_s3_client():
  """Create an S3 client using environment credentials."""
  region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
  kwargs: Dict[str, Any] = {
    "service_name": "s3",
    "region_name": region,
  }
  if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
    kwargs.update(
      aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
      aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )
  return boto3.client(**kwargs)


def _build_dynamo_table(table_name: str):
  """Return a DynamoDB Table resource bound to the configured region."""
  region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
  resource_kwargs: Dict[str, Any] = {}
  if region:
    resource_kwargs["region_name"] = region
  dynamo = boto3.resource("dynamodb", **resource_kwargs)
  return dynamo.Table(table_name)


def _upload_to_s3(s3_client, bucket: str, file_stream: io.BytesIO, filename: str) -> str:
  """Upload the bytes to S3 and return the public URL."""
  file_stream.seek(0)
  extra_args = {"ContentType": "image/jpeg"}
  acl = os.environ.get("AWS_S3_ACL", "public-read").strip()
  if acl:
    extra_args["ACL"] = acl
  s3_client.upload_fileobj(file_stream, bucket, filename, ExtraArgs=extra_args)
  return f"https://{bucket}.s3.amazonaws.com/{filename}"


def _save_to_local_storage(file_bytes: bytes, filename: str) -> Path:
  """Persist uploaded bytes to the local uploads directory."""
  UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
  local_path = UPLOADS_DIR / filename
  with open(local_path, "wb") as destination:
    destination.write(file_bytes)
  return local_path


def _write_temp_file(file_bytes: bytes, filename: str) -> Path:
  """Persist the uploaded bytes temporarily for the local ML fallback."""
  temp_dir = BASE_DIR / "tmp"
  temp_dir.mkdir(exist_ok=True)
  temp_path = temp_dir / filename
  with open(temp_path, "wb") as temp_fp:
    temp_fp.write(file_bytes)
  return temp_path


def _normalise_prediction(raw: Dict[str, Any]) -> Tuple[str, int, List[str], Dict[str, Any]]:
  """
  Ensure the predictor output contains the fields required by the API.

  The ML module currently returns at minimum a `food` label and a calorie
  estimate. We augment this with placeholder ingredients and macro estimates
  when they are missing so the REST contract remains stable.
  """
  food_name = str(raw.get("food") or "Meal")
  calories = int(raw.get("calories") or 0)
  ingredients = raw.get("ingredients") or []
  if not isinstance(ingredients, list):
    ingredients = list(ingredients)
  override_ingredients = _resolve_ingredient_override(food_name)
  if override_ingredients:
    # Prefer curated ingredients when we recognise the meal label.
    ingredients = list(override_ingredients)

  nutrition = raw.get("nutrition_facts")
  if not isinstance(nutrition, dict):
    carbs = round(calories * 0.5 / 4) if calories else 0
    protein = round(calories * 0.25 / 4) if calories else 0
    fat = round(calories * 0.25 / 9) if calories else 0
    nutrition = {
      "calories": calories,
      "carbohydrates": carbs,
      "proteins": protein,
      "fats": fat,
    }
  else:
    nutrition.setdefault("calories", calories)

  return food_name, calories, ingredients, nutrition


def _to_dynamo_compatible(value: Any) -> Any:
  """Convert native Python types into structures acceptable by DynamoDB."""
  if isinstance(value, float):
    return Decimal(str(value))
  if isinstance(value, Decimal):
    return value
  if isinstance(value, dict):
    return {
      str(key): _to_dynamo_compatible(val)
      for key, val in value.items()
      if val is not None
    }
  if isinstance(value, list):
    return [_to_dynamo_compatible(item) for item in value if item is not None]
  return value


def _from_dynamo(value: Any) -> Any:
  """Recursively convert DynamoDB Decimals into JSON friendly primitives."""
  if isinstance(value, Decimal):
    if value % 1 == 0:
      return int(value)
    return float(value)
  if isinstance(value, dict):
    return {key: _from_dynamo(val) for key, val in value.items()}
  if isinstance(value, list):
    return [_from_dynamo(item) for item in value]
  return value


def _get_db_connection() -> sqlite3.Connection:
  """Return a SQLite connection with row access by name."""
  DB_PATH.parent.mkdir(parents=True, exist_ok=True)
  conn = sqlite3.connect(DB_PATH)
  conn.row_factory = sqlite3.Row
  return conn


def _initialise_sqlite() -> None:
  """Ensure the SQLite table exists with the expected schema."""
  conn = _get_db_connection()
  with conn:
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS food_history (
        id TEXT PRIMARY KEY,
        image_url TEXT NOT NULL,
        food TEXT,
        calories INTEGER,
        ingredients TEXT,
        nutrition_facts TEXT,
        metadata TEXT,
        inference_source TEXT,
        ml_service_error TEXT,
        created_at TEXT NOT NULL,
        consumed_at TEXT NOT NULL,
        user_id TEXT
      )
      """
    )

    # Add new columns when migrating from older schemas.
    try:
      conn.execute("ALTER TABLE food_history ADD COLUMN user_id TEXT")
    except sqlite3.OperationalError:
      pass
    try:
      conn.execute(
        "ALTER TABLE food_history ADD COLUMN consumed_at TEXT NOT NULL DEFAULT ''"
      )
    except sqlite3.OperationalError:
      pass
    conn.execute(
      """
      UPDATE food_history
      SET consumed_at = created_at
      WHERE consumed_at IS NULL OR consumed_at = ''
      """
    )
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL
      )
      """
    )
  conn.close()


def _persist_sqlite(record: Dict[str, Any]) -> None:
  """Insert the prediction record into the local SQLite database."""
  conn = _get_db_connection()
  with conn:
    conn.execute(
      """
      INSERT INTO food_history (
        id, user_id, image_url, food, calories, ingredients, nutrition_facts,
        metadata, inference_source, ml_service_error, created_at, consumed_at
      )
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      """,
      (
        record["id"],
        record.get("user_id"),
        record["image_url"],
        record["food"],
        record["calories"],
        json.dumps(record["ingredients"]),
        json.dumps(record["nutrition_facts"]),
        json.dumps(record["metadata"]),
        record["inference_source"],
        record["ml_service_error"],
        record["created_at"],
        record["consumed_at"],
      ),
    )
  conn.close()


def _fetch_sqlite_history(user_id: str | None = None) -> List[Dict[str, Any]]:
  """Return stored prediction history from SQLite."""
  conn = _get_db_connection()
  sql = (
    """
    SELECT id, user_id, image_url, food, calories, ingredients, nutrition_facts,
           metadata, inference_source, ml_service_error, created_at, consumed_at
    FROM food_history
    """
  )
  params: tuple[Any, ...] = ()
  if user_id:
    sql += " WHERE user_id = ?"
    params = (user_id,)
  sql += " ORDER BY datetime(consumed_at) DESC, datetime(created_at) DESC"

  rows = conn.execute(sql, params).fetchall()
  conn.close()

  history: List[Dict[str, Any]] = []
  for row in rows:
    history.append(
      {
        "id": row["id"],
        "user_id": row["user_id"],
        "image_url": row["image_url"],
        "food": row["food"],
        "calories": row["calories"],
        "ingredients": json.loads(row["ingredients"] or "[]"),
        "nutrition_facts": json.loads(row["nutrition_facts"] or "{}"),
        "metadata": json.loads(row["metadata"] or "{}"),
        "inference_source": row["inference_source"],
        "ml_service_error": row["ml_service_error"],
        "created_at": row["created_at"],
        "consumed_at": row["consumed_at"] or row["created_at"],
      }
    )
  return history


def _sqlite_fetch_user(email: str) -> Dict[str, Any] | None:
  """Return a user record from SQLite by email."""
  conn = _get_db_connection()
  row = conn.execute(
    """
    SELECT id, email, password_hash, created_at
    FROM users
    WHERE lower(email) = lower(?)
    """,
    (email,),
  ).fetchone()
  conn.close()

  if row is None:
    return None

  return {
    "id": row["id"],
    "email": row["email"],
    "password_hash": row["password_hash"],
    "created_at": row["created_at"],
  }


def _sqlite_create_user(user_id: str, email: str, password_hash: str, created_at: str) -> None:
  """Persist a new user in SQLite."""
  conn = _get_db_connection()
  with conn:
    conn.execute(
      """
      INSERT INTO users (id, email, password_hash, created_at)
      VALUES (?, ?, ?, ?)
      """,
      (user_id, email, password_hash, created_at),
    )
  conn.close()


def _generate_jwt(user_id: str, email: str) -> str:
  """Return a signed JWT for the provided principal."""
  expiration = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRATION_MINUTES)
  payload = {
    "sub": user_id,
    "email": email,
    "exp": expiration,
    "iat": datetime.now(timezone.utc),
  }
  token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
  if isinstance(token, bytes):  # PyJWT<=1 compatibility
    token = token.decode("utf-8")
  return token


def _decode_jwt(token: str) -> Dict[str, Any]:
  """Decode a JWT and return its payload."""
  return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])


def create_app() -> Flask:
  """Instantiate the Flask application and register routes."""
  app = Flask(__name__)
  CORS(app, resources={r"/*": {"origins": "*"}})

  ml_service_url = os.environ.get("ML_SERVICE_URL", "").strip()
  ml_service_api_key = os.environ.get("ML_SERVICE_API_KEY", "").strip() or None

  def _unauthorized(message: str) -> None:
    response = jsonify({"error": message})
    response.status_code = 401
    abort(response)

  def _get_request_user(optional: bool = True) -> Dict[str, Any] | None:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header or not auth_header.startswith("Bearer "):
      if optional:
        return None
      _unauthorized("Authorization header missing or invalid.")

    token = auth_header.split(" ", 1)[1].strip()
    if not token:
      if optional:
        return None
      _unauthorized("Authorization header missing or invalid.")

    try:
      return _decode_jwt(token)
    except ExpiredSignatureError:
      _unauthorized("Token has expired.")
    except InvalidTokenError:
      _unauthorized("Token is invalid.")
    return None

  def _resolve_consumed_at(raw_value: str | None, default_iso: str) -> str:
    if not raw_value:
      return default_iso
    cleaned = raw_value.strip()
    if not cleaned:
      return default_iso

    try:
      parsed = datetime.fromisoformat(cleaned)
    except ValueError:
      parsed = None

    if parsed is None:
      try:
        parsed_date = datetime.strptime(cleaned, "%Y-%m-%d").date()
        parsed = datetime.combine(parsed_date, datetime.min.time())
      except ValueError:
        app.logger.warning("Invalid meal_date provided; falling back to created_at: %s", cleaned)
        return default_iso

    if parsed.tzinfo is None:
      parsed = parsed.replace(tzinfo=timezone.utc)
    else:
      parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat()

  if USE_AWS_BACKEND:
    s3_bucket = os.environ.get("AWS_BUCKET_NAME")
    if not s3_bucket:
      raise RuntimeError("AWS_BUCKET_NAME must be set when STORAGE_BACKEND=aws.")

    dynamo_table_name = os.environ.get("AWS_DYNAMODB_TABLE")
    if not dynamo_table_name:
      raise RuntimeError("AWS_DYNAMODB_TABLE must be set when STORAGE_BACKEND=aws.")

    dynamo_users_table_name = os.environ.get("AWS_USERS_TABLE")
    if not dynamo_users_table_name:
      raise RuntimeError("AWS_USERS_TABLE must be set when STORAGE_BACKEND=aws.")

    s3_client = _build_s3_client()
    dynamo_table = _build_dynamo_table(dynamo_table_name)
    dynamo_users_table = _build_dynamo_table(dynamo_users_table_name)
  else:
    app.config["UPLOAD_FOLDER"] = str(UPLOADS_DIR)
    _initialise_sqlite()
    s3_client = None
    dynamo_table = None
    s3_bucket = None
    dynamo_users_table = None

  def _get_user_by_email(email: str) -> Dict[str, Any] | None:
    """Fetch user details based on the active storage backend."""
    if USE_AWS_BACKEND:
      assert dynamo_users_table is not None
      response = dynamo_users_table.get_item(Key={"email": email.lower()})
      item = response.get("Item")
      if not item:
        return None
      parsed = _from_dynamo(item)
      parsed["email"] = parsed.get("email", email)
      return parsed

    return _sqlite_fetch_user(email)

  def _create_user(email: str, password: str) -> Dict[str, Any] | None:
    """Persist a new user record via the active storage backend."""
    user_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()
    password_hash = generate_password_hash(password)

    if USE_AWS_BACKEND:
      assert dynamo_users_table is not None
      item = _to_dynamo_compatible(
        {
          "id": user_id,
          "email": email.lower(),
          "password_hash": password_hash,
          "created_at": created_at,
        }
      )
      try:
        dynamo_users_table.put_item(
          Item=item,
          ConditionExpression="attribute_not_exists(email)",
        )
      except ClientError as exc:
        if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
          return None
        raise
    else:
      try:
        _sqlite_create_user(user_id, email.lower(), password_hash, created_at)
      except sqlite3.IntegrityError:
        return None

    return {
      "id": user_id,
      "email": email.lower(),
      "created_at": created_at,
      "password_hash": password_hash,
    }

  @app.route("/predict", methods=["POST"])
  def predict() -> Tuple[Dict[str, Any], int]:
    """
    Handle image uploads, enrich them with metadata, run ML prediction, and persist the result.
    """
    upload_key = "photo" if "photo" in request.files else "image"
    uploaded_file = request.files.get(upload_key)
    if uploaded_file is None or uploaded_file.filename == "":
      return {"error": "No image provided"}, 400

    binary_content = uploaded_file.read()
    if not binary_content:
      return {"error": "Empty file received"}, 400

    extension = Path(uploaded_file.filename).suffix or ".jpg"
    unique_name = f"{uuid.uuid4().hex}{extension.lower()}"

    metadata = extract_metadata(binary_content) or {}
    user_claims = _get_request_user(optional=True)
    user_id = None
    if user_claims:
      user_id = user_claims.get("sub")
      metadata.setdefault("uploaded_by", user_claims.get("email"))
    raw_consumed_at = request.form.get("meal_date") or request.form.get("consumed_at")

    if USE_AWS_BACKEND:
      try:
        assert s3_client is not None and s3_bucket is not None
        image_url = _upload_to_s3(s3_client, s3_bucket, io.BytesIO(binary_content), unique_name)
      except (BotoCoreError, NoCredentialsError, ClientError) as exc:
        app.logger.exception("S3 upload failed: %s", exc)
        return {"error": "Cloud upload failed", "details": str(exc)}, 502
    else:
      local_path = _save_to_local_storage(binary_content, unique_name)
      image_url = urljoin(request.host_url, f"uploads/{local_path.name}")

    hf_prediction: Tuple[str, float, Dict[str, Any]] | None = None
    hf_error: str | None = None
    usda_payload: Dict[str, Any] | None = None
    if HF_SPACE_URL:
      try:
        hf_prediction = _call_hf_food_space(binary_content)
      except HuggingFaceSpaceError as exc:
        hf_error = str(exc)
        app.logger.info("Hugging Face space unavailable: %s", exc)

    raw_prediction: Dict[str, Any] = {}
    inference_source = "huggingface_space" if hf_prediction else "ml_service"
    ml_error: str | None = None

    if hf_prediction:
      hf_label, hf_confidence, hf_payload = hf_prediction
      if hf_confidence < HF_CONFIDENCE_THRESHOLD:
        rejection_payload = {
          "error": "تشخیص و بررسی تصویر ارسالی ممکن نیست",
          "details": f"بالاترین اطمینان {hf_confidence * 100:.1f}% برای {hf_label}",
          "confidence": hf_confidence,
          "label": hf_label,
          "threshold": HF_CONFIDENCE_THRESHOLD,
          "status": "hf_rejected",
          "accepted": False,
          "http_status": HF_REJECTION_STATUS_CODE,
        }
        return rejection_payload, HF_REJECTION_STATUS_CODE

      raw_prediction = predict_calories(unique_name) or {}
      raw_prediction["food"] = hf_label
      raw_prediction["confidence"] = hf_confidence
      metadata["hf_space"] = {
        "label": hf_label,
        "confidence": hf_confidence,
        "threshold": HF_CONFIDENCE_THRESHOLD,
        "url": HF_SPACE_URL,
        "captured_at": datetime.now(timezone.utc).isoformat(),
      }
      for key in ("duration", "average_duration"):
        if isinstance(hf_payload, dict) and key in hf_payload:
          metadata["hf_space"][key] = hf_payload[key]
    else:
      if hf_error:
        metadata["hf_space_error"] = hf_error

      if ml_service_url:
        try:
          raw_prediction = call_ml_service(
            binary_content,
            metadata,
            url=ml_service_url,
            api_key=ml_service_api_key,
          )
        except MLServiceError as exc:
          ml_error = str(exc)
          inference_source = "local_fallback"
          app.logger.warning("ML service call failed; falling back to local predictor: %s", exc)
      else:
        inference_source = "local_fallback"
        ml_error = "ML service URL is not configured."
        app.logger.info("ML service URL missing, using local predictor fallback.")

      if not raw_prediction or "food" not in raw_prediction:
        temp_path = _write_temp_file(binary_content, unique_name)
        try:
          raw_prediction = predict_calories(str(temp_path)) or {}
        except Exception as exc:  # pragma: no cover
          app.logger.exception("Local prediction failed: %s", exc)
          return {"error": "Prediction failed", "details": str(exc)}, 500
        finally:
          temp_path.unlink(missing_ok=True)

    food_name, calories, ingredients, nutrition = _normalise_prediction(raw_prediction)

    if usda_payload is None:
      usda_payload = _aggregate_usda_from_ingredients(food_name, ingredients)
    if usda_payload is None:
      usda_payload = _fetch_usda_nutrition(food_name)
    if usda_payload:
      calories = usda_payload.get("calories") or calories
      usda_macros = usda_payload.get("nutrition_facts") or {}
      if usda_macros:
        usda_macros.setdefault("calories", calories)
        nutrition = usda_macros
      metadata["usda_fdc"] = usda_payload.get("metadata", {})

    created_at = datetime.now(timezone.utc).isoformat()
    consumed_at = _resolve_consumed_at(raw_consumed_at, created_at)
    metadata.setdefault("meal_date", consumed_at)
    record = {
      "id": uuid.uuid4().hex,
      "user_id": user_id,
      "image_url": image_url,
      "food": food_name,
      "calories": calories,
      "ingredients": ingredients,
      "nutrition_facts": nutrition,
      "metadata": metadata,
      "inference_source": inference_source,
      "ml_service_error": ml_error,
      "created_at": created_at,
      "consumed_at": consumed_at,
    }

    if USE_AWS_BACKEND:
      try:
        assert dynamo_table is not None
        dynamo_table.put_item(Item=_to_dynamo_compatible(record))
      except (BotoCoreError, ClientError) as exc:
        app.logger.exception("Failed to persist record to DynamoDB: %s", exc)
        return {"error": "Persistence failed", "details": str(exc)}, 502
    else:
      try:
        _persist_sqlite(record)
      except sqlite3.DatabaseError as exc:
        app.logger.exception("Failed to persist record to SQLite: %s", exc)
        return {"error": "Persistence failed", "details": str(exc)}, 500

    payload = {
      "image_url": image_url,
      "food": food_name,
      "calories": calories,
      "ingredients": ingredients,
      "nutrition_facts": nutrition,
      "metadata": metadata,
      "timestamp": created_at,
      "inference_source": inference_source,
      "consumed_at": consumed_at,
    }
    if user_id:
      payload["user_id"] = user_id
    if ml_error:
      payload["ml_service_error"] = ml_error

    return payload, 200

  @app.route("/auth/signup", methods=["POST"])
  def signup() -> Tuple[Dict[str, Any], int]:
    """Register a new account and issue a JWT."""
    payload = request.get_json(silent=True) or {}
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""

    if not email or not password:
      return {"error": "Email and password are required."}, 400

    if _get_user_by_email(email):
      return {"error": "Email is already registered."}, 409

    user = _create_user(email, password)
    if user is None:
      # Storage layer returned a conflict (duplicate email)
      return {"error": "Email is already registered."}, 409

    token = _generate_jwt(user["id"], user["email"])
    return {
      "token": token,
      "user": {"id": user["id"], "email": user["email"], "created_at": user["created_at"]},
    }, 201

  @app.route("/auth/login", methods=["POST"])
  def login() -> Tuple[Dict[str, Any], int]:
    """Authenticate an existing user and return a JWT."""
    payload = request.get_json(silent=True) or {}
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""

    if not email or not password:
      return {"error": "Email and password are required."}, 400

    user_record = _get_user_by_email(email)
    if not user_record or not check_password_hash(user_record["password_hash"], password):
      return {"error": "Invalid credentials."}, 401

    token = _generate_jwt(user_record["id"], user_record["email"])
    return {
      "token": token,
      "user": {
        "id": user_record["id"],
        "email": user_record["email"],
        "created_at": user_record["created_at"],
      },
    }, 200

  @app.route("/history", methods=["GET"])
  def history() -> Tuple[Dict[str, List[Dict[str, Any]]], int]:
    """Return stored prediction history ordered from newest to oldest."""
    user_claims = _get_request_user(optional=False)
    user_id = user_claims.get("sub") if user_claims else None

    if USE_AWS_BACKEND:
      try:
        assert dynamo_table is not None
        response = dynamo_table.scan()
      except (BotoCoreError, ClientError) as exc:
        app.logger.exception("Failed to read history from DynamoDB: %s", exc)
        return {"error": "History fetch failed", "details": str(exc)}, 502

      items = [_from_dynamo(item) for item in response.get("Items", [])]
      if user_id:
        items = [entry for entry in items if entry.get("user_id") == user_id]
      for entry in items:
        entry.setdefault("consumed_at", entry.get("created_at"))
      items.sort(
        key=lambda entry: (
          entry.get("consumed_at") or entry.get("created_at") or "",
          entry.get("created_at") or "",
        ),
        reverse=True,
      )
    else:
      # Already filtered in SQL when user_id provided.
      items = _fetch_sqlite_history(user_id=user_id)

    seen_ids = set()
    unique_items: List[Dict[str, Any]] = []
    for item in items:
      item_id = item.get("id")
      if item_id and item_id in seen_ids:
        continue
      if item_id:
        seen_ids.add(item_id)
      unique_items.append(item)

    if not unique_items:
      return {"items": [], "message": "history empty"}, 200

    return {"items": unique_items}, 200

  @app.route("/auth/me", methods=["GET"])
  def session() -> Tuple[Dict[str, Any], int]:
    """Return user information for the supplied JWT."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
      return {"error": "Authorization header missing or invalid."}, 401

    token = auth_header.split(" ", 1)[1].strip()
    if not token:
      return {"error": "Authorization header missing or invalid."}, 401

    try:
      claims = _decode_jwt(token)
    except ExpiredSignatureError:
      return {"error": "Token has expired."}, 401
    except InvalidTokenError:
      return {"error": "Token is invalid."}, 401

    email = claims.get("email")
    user_id = claims.get("sub")
    if not email or not user_id:
      return {"error": "Token payload is malformed."}, 401

    user_record = _get_user_by_email(email)
    if not user_record or user_record["id"] != user_id:
      return {"error": "User not found."}, 404

    return {
      "user": {
        "id": user_record["id"],
        "email": user_record["email"],
        "created_at": user_record["created_at"],
      }
    }, 200

  @app.route("/health", methods=["GET"])
  def health() -> Tuple[Dict[str, str], int]:
    """Simple health-check endpoint."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}, 200

  if not USE_AWS_BACKEND:
    @app.route("/uploads/<path:filename>", methods=["GET"])
    def serve_upload(filename: str):
      """Serve locally stored uploads during development."""
      return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

  return app


if __name__ == "__main__":
  flask_app = create_app()
  flask_app.run(host="0.0.0.0", port=5000, debug=True)
