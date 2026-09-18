"""Base detector abstract class."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

from nusagizi.schema import DetectionResult, DetectedIngredient, BoundingBox


class BaseFoodDetector(ABC):
    """Abstract interface for food ingredient detectors."""

    @abstractmethod
    def detect(self, image_path: str, conf_threshold: float = 0.25) -> DetectionResult:
        """Run detection on the given image path."""
        pass

    def annotate_and_save(
        self,
        image_path: str,
        result: DetectionResult,
        output_path: str,
        box_thickness: int = 2,
    ) -> str:
        """Draw bounding boxes and labels on image and save to disk."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")

        h, w, _ = img.shape
        colors = [
            (0, 165, 255),   # Orange (Tempe / Tahu)
            (0, 255, 0),     # Green (Sayur)
            (255, 0, 0),     # Blue (Nasi / Karbohidrat)
            (0, 0, 255),     # Red (Daging / Telur / Ayam)
            (255, 255, 0),   # Cyan
            (255, 0, 255),   # Magenta
        ]

        for i, item in enumerate(result.detected_items):
            box = item.bbox
            x1, y1 = int(box.xmin), int(box.ymin)
            x2, y2 = int(box.xmax), int(box.ymax)

            color = colors[i % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)

            label_text = f"{item.label} ({item.confidence * 100:.1f}%)"
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            # Draw label background
            cv2.rectangle(
                img,
                (x1, max(0, y1 - text_h - baseline - 4)),
                (x1 + text_w + 4, max(text_h + baseline + 4, y1)),
                color,
                -1,
            )
            # Draw label text
            cv2.putText(
                img,
                label_text,
                (x1 + 2, max(text_h, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, img)
        return output_path
