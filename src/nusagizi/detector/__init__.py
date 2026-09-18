"""Detector package for Nusagizi."""

from .base import BaseFoodDetector
from .yolo_world import YOLOWorldDetector
from .custom_yolo import CustomYOLODetector

__all__ = ["BaseFoodDetector", "YOLOWorldDetector", "CustomYOLODetector"]
