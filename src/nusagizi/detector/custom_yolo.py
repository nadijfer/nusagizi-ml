"""Custom Fine-Tuned YOLO Food Detector (Solusi 2)."""

import time
from pathlib import Path
from typing import List, Optional
from ultralytics import YOLO

from nusagizi.detector.base import BaseFoodDetector
from nusagizi.schema import BoundingBox, DetectedIngredient, DetectionResult


class CustomYOLODetector(BaseFoodDetector):
    """Detector using custom trained / fine-tuned YOLO weights (e.g. from Colab)."""

    def __init__(self, weights_path: str):
        """Initialize detector with custom weights.
        
        Args:
            weights_path: Path to custom .pt weights (e.g., 'runs/detect/train/weights/best.pt')
        """
        self.weights_path = weights_path
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Weight file not found: {weights_path}")
        self.model = YOLO(weights_path)

    def detect(self, image_path: str, conf_threshold: float = 0.25) -> DetectionResult:
        """Run inference using custom fine-tuned YOLO model."""
        start_time = time.time()
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            verbose=False,
        )
        inference_time_ms = (time.time() - start_time) * 1000.0

        detected_items: List[DetectedIngredient] = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            names = results[0].names
            for box in boxes:
                cls_idx = int(box.cls[0].item())
                label = names.get(cls_idx, f"class_{cls_idx}")
                conf = float(box.conf[0].item())
                coords = box.xyxy[0].tolist()

                detected_items.append(
                    DetectedIngredient(
                        label=label,
                        confidence=conf,
                        bbox=BoundingBox(
                            xmin=coords[0],
                            ymin=coords[1],
                            xmax=coords[2],
                            ymax=coords[3],
                        ),
                    )
                )

        return DetectionResult(
            image_path=image_path,
            model_name=f"Custom YOLO ({Path(self.weights_path).name})",
            detected_items=detected_items,
            inference_time_ms=inference_time_ms,
        )
