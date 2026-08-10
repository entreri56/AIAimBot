"""
Debug Viewer — Capture Card Edition
Captures directly from a USB capture card (e.g. USB3.0 Capture) via DirectShow.
Bypasses OBS entirely — reads raw video from the capture card hardware.

Usage:
    # List all DirectShow devices (find your capture card)
    python debug_viewer_capture_card.py --list-devices

    # Run with auto-detection (tries camera 0, then 1, then 2...)
    python debug_viewer_capture_card.py

    # Run with specific camera index
    python debug_viewer_capture_card.py --camera 1

    # Full options
    python debug_viewer_capture_card.py --camera 0 --conf 0.3 --head-class 0

Controls:
    ↑ / ↓        — Adjust confidence threshold
    H             — Toggle head-class filter on/off (show all detections)
    L             — Toggle labels (confidence text)
    A             — Toggle aim-point dots
    C             — Toggle crosshair
    Space         — Pause / resume
    S             — Save current frame to disk
    ESC / Q       — Quit
"""

import cv2
import sys
import os
import time
import argparse
import threading
import numpy as np


# ── Parse command-line args ───────────────────────────────────────────
parser = argparse.ArgumentParser(description="YoloAimBot Debug Viewer — Capture Card Edition")
parser.add_argument("--model", default="models/best.pt", help="Path to YOLO model")
parser.add_argument("--camera", type=int, default=-1, help="Camera device index (-1 = auto-detect)")
parser.add_argument("--width", type=int, default=1920, help="Requested capture width")
parser.add_argument("--height", type=int, default=1080, help="Requested capture height")
parser.add_argument("--conf", type=float, default=0.3, help="Initial confidence threshold")
parser.add_argument("--head-class", type=int, default=0, help="Class ID for head/player targets")
parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
parser.add_argument("--list-devices", action="store_true", help="List all DirectShow devices and exit")
args = parser.parse_args()


# ── List devices mode ─────────────────────────────────────────────────
def list_dshow_devices():
    """Enumerate all DirectShow video capture devices with detailed info."""
    print("=" * 70)
    print("  Scanning DirectShow video capture devices...")
    print("=" * 70)

    devices = []
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            # Get backend name
            backend = cap.getBackendName() if hasattr(cap, 'getBackendName') else "DSHOW"

            # Try to read a frame to verify the device actually works
            ret, frame = cap.read()
            frame_ok = ret and frame is not None

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Try to get the friendly name via DirectShow
            friendly_name = f"Camera {i}"
            try:
                # Query DirectShow for the device name
                cap.set(cv2.CAP_PROP_SETTINGS, 0)  # This can trigger the property page
            except Exception:
                pass

            devices.append({
                "index": i,
                "width": w,
                "height": h,
                "fps": fps,
                "backend": backend,
                "frame_ok": frame_ok,
                "name": friendly_name,
            })
            cap.release()
        else:
            cap.release()

    if not devices:
        print("\n  No DirectShow devices found!")
        print("  Make sure your USB capture card is plugged in.")
        print("  Try: python debug_viewer_capture_card.py --list-devices")
        return []

    print(f"\n  Found {len(devices)} device(s):\n")
    for d in devices:
        status = "✓ (frame grab OK)" if d["frame_ok"] else "✗ (can't grab frames)"
        print(f"  [{d['index']}] {d['width']}x{d['height']} @ {d['fps']:.0f}fps  {status}")
        print(f"      Backend: {d['backend']}")

    print(f"\n  Tip: Your USB3.0 capture card is likely one of the devices above.")
    print(f"  If a device shows 'can't grab frames', it may be in use by OBS.")
    print(f"  Close OBS or stop the Video Capture Device source, then try again.")
    print(f"\n  Usage: python debug_viewer_capture_card.py --camera <index>")
    return devices


if args.list_devices:
    list_dshow_devices()
    sys.exit(0)


# ── Load model ────────────────────────────────────────────────────────
from detection import load_model

print(f"[INFO] Loading model: {args.model}")
if not os.path.exists(args.model):
    print(f"[ERROR] Model file not found: {args.model}")
    sys.exit(1)

model, class_names = load_model(args.model)
print(f"[INFO] Model loaded. Classes: {class_names}")
head_class_name = class_names.get(args.head_class, f"class_{args.head_class}")
print(f"[INFO] Head class ID {args.head_class} -> '{head_class_name}'")


# ── Open capture card ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  Opening capture card...")
print("=" * 70)

cap = None
cap_w = args.width
cap_h = args.height

# Determine which camera indices to try
if args.camera >= 0:
    indices_to_try = [args.camera]
else:
    # Auto-detect: try indices 0-5
    indices_to_try = list(range(6))

opened_index = -1
for idx in indices_to_try:
    print(f"[INFO] Trying camera {idx} with DirectShow...")
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = None
        print(f"       Camera {idx}: not available")
        continue

    # Configure resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Test grab a frame
    ret, test_frame = cap.read()
    if ret and test_frame is not None:
        cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[INFO] SUCCESS: Camera {idx} — {cap_w}x{cap_h} @ {actual_fps:.0f}fps")
        opened_index = idx
        break
    else:
        print(f"       Camera {idx}: opened but can't grab frames (in use by OBS?)")
        cap.release()
        cap = None

if cap is None:
    print("\n[ERROR] Could not open any capture card.")
    print("        Troubleshooting:")
    print("        1. Is the USB capture card plugged in?")
    print("        2. Is OBS using the device? Close OBS or disable the source.")
    print("        3. List devices: python debug_viewer_capture_card.py --list-devices")
    print("        4. Try specific index: python debug_viewer_capture_card.py --camera 1")
    sys.exit(1)


# ── Threaded frame grabber (keeps the DSHOW graph alive) ──────────────
_frame_lock = threading.Lock()
_latest_frame = test_frame  # seed with the test frame
_running = True

def capture_loop():
    """Dedicated thread that continuously grabs frames for lowest latency."""
    global _latest_frame
    while _running:
        ret, frame = cap.read()
        if ret and frame is not None:
            with _frame_lock:
                _latest_frame = frame
        else:
            time.sleep(0.001)

capture_thread = threading.Thread(target=capture_loop, daemon=True)
capture_thread.start()
print(f"[INFO] Capture thread started (camera {opened_index})")


def grab_frame():
    """Get the most recent frame (non-blocking)."""
    with _frame_lock:
        if _latest_frame is None:
            return None
        return _latest_frame.copy()


# ── State ─────────────────────────────────────────────────────────────
conf_threshold = args.conf
show_all_classes = False
show_labels = True
show_aim_points = True
show_crosshair = True
paused = False
running = True

# FPS tracking
fps = 0.0
frame_count = 0
fps_timer = time.perf_counter()

# Color map per class
CLASS_COLORS = {}
def get_class_color(class_id):
    if class_id not in CLASS_COLORS:
        np.random.seed(class_id * 31 + 7)
        CLASS_COLORS[class_id] = tuple(int(c) for c in np.random.randint(64, 255, 3))
    return CLASS_COLORS[class_id]


# ── Drawing function ──────────────────────────────────────────────────
def draw_overlay(frame, boxes_data, crosshair, head_cls, head_name):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    if show_crosshair and crosshair:
        cv2.drawMarker(overlay, crosshair, (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.drawMarker(overlay, crosshair, (0, 0, 0), cv2.MARKER_CROSS, 24, 1)

    for box in boxes_data:
        cls_id = box["cls"]
        is_head = (cls_id == head_cls)
        cls_name = class_names.get(cls_id, f"cls_{cls_id}")

        if is_head:
            color = (0, 255, 0)
            thickness = 3
        else:
            color = get_class_color(cls_id)
            color = tuple(int(c * 0.6) for c in color)
            thickness = 1

        x1, y1 = int(box["x1"]), int(box["y1"])
        x2, y2 = int(box["x2"]), int(box["y2"])

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

        if show_aim_points and is_head:
            ax, ay = int(box["aim_x"]), int(box["aim_y"])
            cv2.circle(overlay, (ax, ay), 5, (0, 255, 255), -1)
            cv2.circle(overlay, (ax, ay), 7, (0, 0, 0), 1)

        if show_labels:
            label = f"{cls_name} {box['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(overlay, label, (x1 + 3, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # HUD panel
    hud_lines = [
        f"Capture Card | FPS: {fps:.1f}",
        f"Detections: {len(boxes_data)} (filter: {'OFF' if show_all_classes else 'ON'})",
        f"Conf Threshold: {conf_threshold:.2f}  [Up/Down arrows]",
        f"Head Class: {head_name} (ID {head_cls})",
        f"Labels: {'ON' if show_labels else 'OFF'}  [L] | Aim Pts: {'ON' if show_aim_points else 'OFF'}  [A]",
        f"Paused: {'YES' if paused else 'NO'}  [Space]",
    ]
    panel_x, panel_y = 12, 12
    line_h = 20
    panel_w = 420
    panel_h = len(hud_lines) * line_h + 16

    roi = overlay[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w]
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
    blended = cv2.addWeighted(roi, 0.4, overlay[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w], 0.6, 0)
    overlay[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w] = blended

    for i, line in enumerate(hud_lines):
        y = panel_y + 18 + i * line_h
        cv2.putText(overlay, line, (panel_x + 8, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

    # Bottom bar
    bar_text = f" Q/ESC:Quit | S:Save Frame | H:Toggle Classes | Capture Card {cap_w}x{cap_h}"
    cv2.rectangle(overlay, (0, h - 26), (w, h), (15, 15, 15), -1)
    cv2.putText(overlay, bar_text, (10, h - 8),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # Confidence bar (right edge)
    bar_x = w - 30
    bar_h = h - 80
    bar_y = 40
    cv2.rectangle(overlay, (bar_x - 6, bar_y - 4), (bar_x + 14, bar_y + bar_h + 4), (30, 30, 30), -1)
    cv2.rectangle(overlay, (bar_x - 6, bar_y - 4), (bar_x + 14, bar_y + bar_h + 4), (80, 80, 80), 1)

    fill_h = int(bar_h * conf_threshold)
    if conf_threshold < 0.5:
        ratio = conf_threshold / 0.5
        bar_color = (0, int(255 * ratio), int(255 * (1 - ratio)))
    else:
        ratio = (conf_threshold - 0.5) / 0.5
        bar_color = (0, 255, int(255 * (1 - ratio)))
    cv2.rectangle(overlay, (bar_x - 4, bar_y + bar_h - fill_h), (bar_x + 12, bar_y + bar_h), bar_color, -1)
    cv2.putText(overlay, f"{conf_threshold:.2f}", (bar_x - 20, bar_y + bar_h + 18),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return overlay


# ── Run inference on a frame ──────────────────────────────────────────
def run_inference(frame):
    """Run YOLO on a frame, return list of parsed detection dicts."""
    results = list(model.predict(
        source=frame,
        imgsz=args.imgsz,
        conf=conf_threshold,
        iou=0.5,
        max_det=50,
        stream=True,
        verbose=False,
        show=False,
    ))

    boxes_data = []
    for result in results:
        if result.boxes is None:
            continue
        boxes = result.boxes
        cls_arr = boxes.cls.cpu().numpy()
        xyxy_arr = boxes.xyxy.cpu().numpy()
        confs_arr = boxes.conf.cpu().numpy()

        for i in range(len(cls_arr)):
            class_id = int(cls_arr[i])
            confidence = float(confs_arr[i])

            if not show_all_classes and class_id != args.head_class:
                continue

            x1, y1, x2, y2 = xyxy_arr[i]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            aim_x = cx
            aim_y = y1 + (y2 - y1) * 0.15

            boxes_data.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "conf": confidence,
                "cls": class_id,
                "cx": cx, "cy": cy,
                "aim_x": aim_x, "aim_y": aim_y,
            })

    # Print detections to console
    for box in boxes_data:
        cls_name = class_names.get(box["cls"], f"cls_{box['cls']}")
        print(f"  [DETECT] {cls_name} | conf={box['conf']:.3f} | "
              f"pos=({box['cx']:.0f},{box['cy']:.0f}) | "
              f"size=({box['x2']-box['x1']:.0f}x{box['y2']-box['y1']:.0f})")

    return boxes_data


# ── Main loop ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  Debug Viewer Running — Capture Card Direct")
print("  Press ESC or Q to quit. See HUD panel for all controls.")
print("=" * 70)

last_console_print = 0.0
last_frame = None

try:
    while running:
        if not paused:
            frame = grab_frame()
            if frame is None:
                time.sleep(0.005)
                continue
            last_frame = frame.copy()
        else:
            if last_frame is None:
                time.sleep(0.01)
                continue
            frame = last_frame.copy()

        t0 = time.perf_counter()
        boxes_data = run_inference(frame)
        inference_ms = (time.perf_counter() - t0) * 1000

        # Throttled console output
        now = time.perf_counter()
        if now - last_console_print > 1.0:
            n_head = sum(1 for b in boxes_data if b["cls"] == args.head_class)
            n_other = len(boxes_data) - n_head
            print(f"[STATUS] Detections: {len(boxes_data)} total "
                  f"({n_head} head, {n_other} other) | "
                  f"conf={conf_threshold:.2f} | "
                  f"inference={inference_ms:.1f}ms | fps={fps:.1f}")
            last_console_print = now

        # Draw
        display = draw_overlay(frame, boxes_data,
                               crosshair=(cap_w // 2, cap_h // 2),
                               head_cls=args.head_class,
                               head_name=head_class_name)

        # Resize if too wide
        dh, dw = display.shape[:2]
        if dw > 1600:
            scale = 1600 / dw
            display = cv2.resize(display, (1600, int(dh * scale)))

        cv2.imshow("YoloAimBot — Capture Card Debug Viewer", display)

        # FPS
        frame_count += 1
        if now - fps_timer >= 1.0:
            fps = frame_count / (now - fps_timer)
            frame_count = 0
            fps_timer = now

        # Keyboard
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord('q'):
            running = False
        elif key == ord(' '):
            paused = not paused
            print(f"[INFO] {'PAUSED' if paused else 'RESUMED'}")
        elif key == ord('h') or key == ord('H'):
            show_all_classes = not show_all_classes
            print(f"[INFO] Show all classes: {show_all_classes}")
        elif key == ord('l') or key == ord('L'):
            show_labels = not show_labels
        elif key == ord('a') or key == ord('A'):
            show_aim_points = not show_aim_points
        elif key == ord('c') or key == ord('C'):
            show_crosshair = not show_crosshair
        elif key == ord('s') or key == ord('S'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"debug_capture_{timestamp}.png"
            cv2.imwrite(filename, display)
            print(f"[SAVE] Frame saved: {filename}")
        elif key == 82:   # Up arrow
            conf_threshold = min(0.99, conf_threshold + 0.05)
            print(f"[INFO] Confidence threshold: {conf_threshold:.2f}")
        elif key == 84:   # Down arrow
            conf_threshold = max(0.05, conf_threshold - 0.05)
            print(f"[INFO] Confidence threshold: {conf_threshold:.2f}")

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user.")

finally:
    _running = False
    capture_thread.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Debug viewer closed.")