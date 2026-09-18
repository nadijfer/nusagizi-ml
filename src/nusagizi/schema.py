"""Schema and data structures for Nusagizi Food Detection."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BoundingBox:
    """Bounding box coordinates in pixels [xmin, ymin, xmax, ymax]."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_list(self) -> List[float]:
        return [round(self.xmin, 2), round(self.ymin, 2), round(self.xmax, 2), round(self.ymax, 2)]


@dataclass
class DetectedIngredient:
    """Representing a single detected food item/ingredient."""
    label: str
    confidence: float
    bbox: BoundingBox
    category: Optional[str] = None  # e.g., "protein_nabati", "protein_hewani", "karbohidrat", "sayur"

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "category": self.category,
            "bbox": self.bbox.to_list(),
        }


@dataclass
class DetectionResult:
    """Complete detection result for an image."""
    image_path: str
    model_name: str
    detected_items: List[DetectedIngredient] = field(default_factory=list)
    inference_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "model_name": self.model_name,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "total_items_detected": len(self.detected_items),
            "items": [item.to_dict() for item in self.detected_items],
        }
