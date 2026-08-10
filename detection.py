"""
Detection module for YoloAimBot - Capture Card Edition.
Handles YOLO model loading and inference.
"""

import numpy as np
from ultralytics import YOLO

# Global model cache
_model = None
_model_path = None
_class_names = {}


def load_model(model_path):
    """
    Load a YOLO model. Caches the model to avoid reloading.
    Returns (model, class_names_dict).
    """
    global _model, _model_path, _class_names

    if _model is not None and _model_path == model_path:
        return _model, _class_names

    print(f"[MODEL] Loading: {model_path}")
    _model = YOLO(model_path, task="detect")

    # Extract class names
    if hasattr(_model, "names"):
        _class_names = _model.names
    elif hasattr(_model.model, "names"):
        _class_names = _model.model.names
    else:
        _class_names = {}

    _model_path = model_path
    print(f"[MODEL] Loaded. Classes: {_class_names}")
    return _model, _class_names


def detect(model, image, conf=0.3, imgsz=640, max_det=50, verbose=False):
    """
    Run YOLO detection on an image.
    Returns list of results (ultralytics Results objects).
    """
    results = model.predict(
        source=image,
        imgsz=imgsz,
        conf=conf,
        iou=0.5,
        max_det=max_det,
        stream=True,
        verbose=verbose,
        show=False,
    )
    return results


def extract_targets(results, class_names, head_class, min_conf=0.3, aim_y_fraction=0.15, head_classes=None):
    """
    Extract target candidates from detection results.

    Args:
        results: YOLO detection results
        class_names: Dict of {class_id: class_name}
        head_class: Primary class ID to target (used if head_classes is None)
        min_conf: Minimum confidence threshold
        aim_y_fraction: Where on the box to aim (0=top, 0.5=center)
        head_classes: Optional list of class IDs to target (e.g. [0, 1] for CT+T).
                      If provided, overrides head_class.

    Returns:
        List of dicts: [{x, y, dist, conf, cls, cx, cy, x1, y1, x2, y2}, ...]
    """
    targets = []

    # Determine which class IDs to accept
    if head_classes is not None and len(head_classes) > 0:
        valid_classes = set(head_classes)
    else:
        valid_classes = {head_class}

    for result in results:
        if result.boxes is None:
            continue

        boxes = result.boxes
        cls = boxes.cls.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        # Filter by class and confidence
        for i in range(len(cls)):
            class_id = int(cls[i])
            if class_id not in valid_classes:
                continue
            if float(confs[i]) < min_conf:
                continue

            x1, y1, x2, y2 = xyxy[i]
            conf = float(confs[i])

            # Aim point: center X, adjustable Y
            aim_x = (x1 + x2) / 2.0
            aim_y = y1 + (y2 - y1) * aim_y_fraction

            # Box center (for tracking)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            targets.append({
                "x": aim_x,
                "y": aim_y,
                "dist": 0.0,  # Will be calculated relative to crosshair
                "conf": conf,
                "cls": class_id,
                "cx": cx,
                "cy": cy,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            })

    return targets


def draw_debug_overlay(image, targets, locked_target=None, crosshair=None, 
                       head_class_name="head", show_conf=True):
    """
    Draw detection boxes and info on an image for debugging.
    Returns the annotated image.
    """
    import cv2

    debug = image.copy()
    h, w = debug.shape[:2]

    # Draw crosshair
    if crosshair is None:
        crosshair = (w // 2, h // 2)
    cv2.drawMarker(debug, crosshair, (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

    # Draw targets
    for t in targets:
        x1, y1, x2, y2 = int(t["x1"]), int(t["y1"]), int(t["x2"]), int(t["y2"])
        ax, ay = int(t["x"]), int(t["y"])

        # Green box for all targets
        color = (0, 255, 0)
        thickness = 2

        # Highlight locked target in red
        if locked_target and t["cx"] == locked_target[0] and t["cy"] == locked_target[1]:
            color = (0, 0, 255)
            thickness = 3

        cv2.rectangle(debug, (x1, y1), (x2, y2), color, thickness)

        # Draw aim point
        cv2.circle(debug, (ax, ay), 3, (0, 255, 255), -1)

        # Label
        if show_conf:
            label = f"{head_class_name} {t['conf']:.2f}"
            cv2.putText(debug, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Status text
    status = f"Targets: {len(targets)}"
    cv2.putText(debug, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return debug