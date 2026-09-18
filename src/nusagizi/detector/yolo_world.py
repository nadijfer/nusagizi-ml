"""YOLO-World Zero-Shot Open-Vocabulary Food Detector."""

import time
from typing import List, Optional, Dict, Tuple
from ultralytics import YOLOWorld

from nusagizi.detector.base import BaseFoodDetector
from nusagizi.schema import BoundingBox, DetectedIngredient, DetectionResult


class YOLOWorldDetector(BaseFoodDetector):
    """Zero-Shot Open-Vocabulary Food Detector using YOLO-World.
    
    Requires NO dataset annotation and NO training.
    Classes can be defined dynamically on the fly.
    """

    DEFAULT_CLASSES = [
        "tempeh",
        "fried tempeh",
        "tofu",
        "fried tofu",
        "boiled egg",
        "fried egg",
        "steamed rice",
        "porridge",
        "fried chicken",
        "chicken",
        "fish",
        "vegetables",
        "spinach",
        "carrot",
        "banana",
        "papaya",
    ]

    # Indonesian label & Kemenkes food category mappings
    METADATA_MAP: Dict[str, Tuple[str, str]] = {
        "tempeh": ("Tempe", "protein_nabati"),
        "fried tempeh": ("Tempe Goreng", "protein_nabati"),
        "tofu": ("Tahu", "protein_nabati"),
        "fried tofu": ("Tahu Goreng", "protein_nabati"),
        "boiled egg": ("Telur Rebus", "protein_hewani"),
        "fried egg": ("Telur Dadar / Ceplok", "protein_hewani"),
        "steamed rice": ("Nasi Putih", "karbohidrat"),
        "porridge": ("Bubur", "karbohidrat"),
        "fried chicken": ("Ayam Goreng", "protein_hewani"),
        "chicken": ("Ayam", "protein_hewani"),
        "fish": ("Ikan", "protein_hewani"),
        "vegetables": ("Sayuran", "sayur"),
        "spinach": ("Bayam", "sayur"),
        "carrot": ("Wortel", "sayur"),
        "banana": ("Pisang", "buah"),
        "papaya": ("Pepaya", "buah"),
    }

    def __init__(
        self,
        model_name: str = "yolov8s-worldv2.pt",
        custom_classes: Optional[List[str]] = None,
    ):
        """Initialize YOLO-World model.
        
        Args:
            model_name: Ultralytics YOLO-World checkpoint (e.g. 'yolov8s-worldv2.pt')
            custom_classes: Optional list of text class prompts.
        """
        self.model_name = model_name
        self.model = YOLOWorld(model_name)
        self.classes = custom_classes or self.DEFAULT_CLASSES
        self.update_classes(self.classes)

    def update_classes(self, new_classes: List[str]) -> None:
        """Update active vocabulary classes on-the-fly without retraining."""
        self.classes = new_classes
        self.model.set_classes(self.classes)

    def detect(self, image_path: str, conf_threshold: float = 0.20) -> DetectionResult:
        """Detect food items in the specified image.
        
        Args:
            image_path: Path to the food image file.
            conf_threshold: Minimum confidence threshold (default 0.20 for zero-shot).
            
        Returns:
            DetectionResult with detected items, confidence, and bounding boxes.
        """
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
            for box in boxes:
                cls_idx = int(box.cls[0].item())
                raw_label = self.classes[cls_idx] if cls_idx < len(self.classes) else f"class_{cls_idx}"
                conf = float(box.conf[0].item())
                coords = box.xyxy[0].tolist()

                # Map to Indonesian label & category if known
                display_label, category = self.METADATA_MAP.get(
                    raw_label.lower(), (raw_label, "lainnya")
                )

                detected_items.append(
                    DetectedIngredient(
                        label=display_label,
                        confidence=conf,
                        bbox=BoundingBox(
                            xmin=coords[0],
                            ymin=coords[1],
                            xmax=coords[2],
                            ymax=coords[3],
                        ),
                        category=category,
                    )
                )

        return DetectionResult(
            image_path=image_path,
            model_name=f"YOLO-World ({self.model_name})",
            detected_items=detected_items,
            inference_time_ms=inference_time_ms,
        )
