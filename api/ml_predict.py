"""Lightweight helpers for generating meal predictions in development.

The original implementation relied on TensorFlow and a custom classifier. To
keep deployments small and fast we now return synthetic predictions that mimic
the shape of the production response.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict

logger = logging.getLogger(__name__)


_MEAL_TEMPLATES = [
  {
    "food": "Grilled Chicken Salad",
    "calories": 420,
    "ingredients": [
      "Grilled Chicken Breast",
      "Mixed Greens",
      "Cherry Tomatoes",
      "Cucumber",
      "Balsamic Vinaigrette",
    ],
    "macros": {"protein": 38, "carbohydrates": 18, "fats": 18},
  },
  {
    "food": "Veggie Power Bowl",
    "calories": 510,
    "ingredients": [
      "Quinoa",
      "Roasted Sweet Potato",
      "Black Beans",
      "Avocado",
      "Lime Dressing",
    ],
    "macros": {"protein": 21, "carbohydrates": 62, "fats": 20},
  },
  {
    "food": "Salmon Sushi Roll",
    "calories": 360,
    "ingredients": [
      "Sushi Rice",
      "Fresh Salmon",
      "Nori",
      "Cucumber",
      "Soy Sauce",
    ],
    "macros": {"protein": 24, "carbohydrates": 44, "fats": 10},
  },
  {
    "food": "Mediterranean Wrap",
    "calories": 480,
    "ingredients": [
      "Whole Wheat Tortilla",
      "Hummus",
      "Feta Cheese",
      "Kalamata Olives",
      "Spinach",
    ],
    "macros": {"protein": 19, "carbohydrates": 48, "fats": 22},
  },
  {
    "food": "Spaghetti Bolognese",
    "calories": 610,
    "ingredients": [
      "Spaghetti",
      "Tomato Sauce",
      "Ground Beef",
      "Parmesan Cheese",
      "Basil",
    ],
    "macros": {"protein": 32, "carbohydrates": 68, "fats": 22},
  },
]


def _synthetic_prediction() -> Dict[str, Any]:
  """Return a deterministic-but-randomised meal prediction."""
  template = random.choice(_MEAL_TEMPLATES)
  jitter = random.randint(-40, 60)
  calories = max(120, template["calories"] + jitter)
  scale = calories / template["calories"]
  nutrition = {
    "calories": calories,
    "proteins": max(1, int(round(template["macros"]["protein"] * scale))),
    "carbohydrates": max(1, int(round(template["macros"]["carbohydrates"] * scale))),
    "fats": max(1, int(round(template["macros"]["fats"] * scale))),
  }
  return {
    "food": template["food"],
    "calories": calories,
    "ingredients": list(template["ingredients"]),
    "nutrition_facts": nutrition,
    "confidence": round(random.uniform(0.72, 0.94), 2),
  }


def predict_calories(image_path: str) -> Dict[str, Any]:
  """
  Return a prediction for the supplied image.

  ``image_path`` is accepted solely to preserve the public API – the result is
  currently independent of the image contents.
  """
  logger.info("TensorFlow disabled; returning synthetic prediction for %s", image_path)
  return _synthetic_prediction()


__all__ = ["predict_calories"]
