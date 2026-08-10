"""Check class names for all models in the models/ folder."""
import os
from ultralytics import YOLO

model_dir = "models"
for f in sorted(os.listdir(model_dir)):
    if f.endswith((".pt", ".onnx")):
        path = os.path.join(model_dir, f)
        try:
            model = YOLO(path, task="detect")
            print(f"{f}: {model.names}")
        except Exception as e:
            print(f"{f}: ERROR — {e}")