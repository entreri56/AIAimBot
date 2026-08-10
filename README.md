# 🎯 YoloAimBot — Capture Card Edition

> **⚠️ EDUCATIONAL PURPOSES ONLY**
>
> This project is intended **solely for educational and research purposes** — to explore computer vision, real-time object detection with YOLO, and hardware-in-the-loop systems. **It is not intended for use in competitive online games.** Using this software in multiplayer games may violate the game's Terms of Service and result in permanent bans. I assume no liability for misuse.

---

## 📋 Overview

YoloAimBot is an AI-powered aim-assist tool built around **YOLO (You Only Look Once)** object detection. It runs on a **separate PC** (not the gaming PC) to avoid detection by anti-cheat systems.

```
┌─────────────┐    HDMI    ┌──────────────┐    DirectShow/USB    ┌──────────────┐      USB      ┌─────────────┐
│  Gaming PC  │ ─────────→ │ Capture Card │ ───────────────────→ │  This Bot    │ ────────────→ │   Gaming PC │
│  (CS2/etc)  │            │              │                      │  (Python)    │    MAKCU      │  (mouse in) │
└─────────────┘            └──────────────┘                      └──────────────┘               └─────────────┘
```

1. **Gaming PC** outputs video over HDMI
2. **Capture Card** plugs into the 2nd PC via USB — appears as a DirectShow camera device
3. **This program** reads frames directly from the capture card via OpenCV (`cv2.VideoCapture` + `CAP_DSHOW`) — **no OBS required**
4. YOLO detection runs on each frame to find targets and calculate mouse movements
5. **MAKCU** (Mouse And Keyboard Controlled via USB) sends movements back to the Gaming PC

---

## 🔌 Hardware Requirements

### **You MUST have a capture card — this is non-negotiable.**

| Component | Purpose |
|---|---|
| **Capture card** | Captures HDMI output from gaming PC (e.g., Elgato HD60 X, AverMedia Live Gamer, or generic USB 3.0 HDMI capture dongle). Plugs into the 2nd PC via USB and appears as a DirectShow camera — read directly by OpenCV. |
| **Second PC** | Runs this Python program — does NOT need a powerful GPU |
| **MAKCU device** | Hardware mouse emulator (Arduino/Teensy-based USB HID device) that connects via USB to the gaming PC |
| **HDMI cable(s)** | To connect the gaming PC to the capture card |

### Recommended capture cards:
- Elgato HD60 X / 4K X
- AverMedia Live Gamer Ultra
- Generic USB 3.0 HDMI Video Capture dongles ($15–$30 on Amazon/AliExpress)

### Do I need OBS?
**No.** The `opencv` capture mode reads the capture card directly via DirectShow — OBS is not required. OBS is only needed if you want to use the `ndi` capture mode (lowest latency), or if you prefer to route video through OBS Virtual Camera (e.g., for overlays or scene compositing).

### Why a capture card?
- Anti-cheat software cannot detect video capture happening on a **different physical machine**
- The capture card appears as a standard DirectShow camera — OpenCV reads it directly with `cv2.VideoCapture()`
---

## 🚀 Quick Start

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Plug in your capture card

1. Connect the **Gaming PC's HDMI out** → **Capture Card HDMI in**
2. Plug the **Capture Card's USB** → **2nd PC**
3. Verify it's detected:

```bash
python -c "import cv2; cap=cv2.VideoCapture(0,cv2.CAP_DSHOW); print('OK' if cap.isOpened() else 'FAIL'); cap.release()"
```

Or use the built-in debug viewer to list devices:
```bash
python debug_viewer_capture_card.py --list-devices
```

### 3. Train a YOLO model (or download one)

You need a trained YOLO model that can detect players/heads in your game. Place the final `.pt` file in the `models/` folder:

```
models/
  └── best.pt        ← your trained YOLO model
```

#### Training workflow:

1. **Capture screenshots** from your game (via capture card). Aim for 640×640 resolution — this matches the YOLO inference size for best speed/accuracy balance. 500–2000 varied images is a good starting point.

2. **Label your images** with [LabelImg](https://github.com/HumanSignal/labelImg) (free, open-source):
   ```bash
   pip install labelImg
   labelImg
   ```
   - Open your screenshot folder
   - Draw bounding boxes around players/heads
   - Save in **YOLO format** (`.txt` files with class_id x_center y_center width height)
   - Example class setup: `0` = CT/Terrorist head, `1` = CT/Terrorist body

   Example annotated 640×640 frame:
   <img width="640" height="640" alt="capture_0003_2026-07-24_21-26-53-382" src="https://github.com/user-attachments/assets/1c8bd4f8-fb39-4fcc-8a12-84a0193f0c53" />


3. **Train with YOLOv11** (lightweight, fast inference):
   ```python
   from ultralytics import YOLO

   # Load a pretrained nano/small model (fast, low resource usage)
   model = YOLO("yolo11n.pt")  # or yolo11s.pt for slightly better accuracy

   # Train
   model.train(
       data="dataset/data.yaml",   # path to your dataset config
       epochs=100,
       imgsz=640,
       batch=16,
       device="cpu",               # or "cuda" / "mps"
   )

   # Export
   model.export(format="onnx")     # optional: faster inference
   ```

   Copy the trained `best.pt` (from `runs/detect/train/weights/`) into `models/`.

4. **Verify** your model's classes:
   ```bash
   python check_models.py
   ```

> **Tip:** YOLOv11n (nano) runs fast on CPU — ideal for the 2nd PC. If you have a GPU on the 2nd PC, use `yolo11s.pt` or `yolo11m.pt` for better accuracy. Any YOLOv5/v8/v10 model also works.

### 4. (Optional) Set up OBS

OBS is **not required** — the bot reads the capture card directly. Only set this up if you need:
- **NDI mode** — for the lowest possible latency
- **Virtual Camera** — if you want OBS overlays, scene compositing, or filters on the feed

If using OBS:
1. Install [OBS Studio](https://obsproject.com/)
2. Add your capture card as a Video Capture Device source
3. Start **Virtual Camera** (Tools → Virtual Camera) or install the [OBS-NDI plugin](https://github.com/obs-ndi/obs-ndi)

### 5. Configure

Edit `config.py` or use the GUI:

```python
# config.py key settings:
CAPTURE_MODE  = "opencv"     # "opencv", "ndi", or "mss" (single-PC fallback)
CAMERA_INDEX  = 0            # Capture card device index (use --list-devices to find)
IN_GAME_SENS  = 1.25         # Match your in-game sensitivity
HEAD_CLASSES  = [0, 1]       # Target class IDs (CS2: 0=CT, 1=T)
```

### 6. Run

```bash
# GUI mode (recommended):
python gui.py

# Or double-click:
run.bat
```

---

## 🧵 Architecture

| Thread | Purpose |
|---|---|
| **Detection Thread** | Captures frames → runs YOLO inference → finds best target → writes to shared state |
| **Movement Thread** | Reads shared target state → applies smoothing/deadzone → sends movement commands via MAKCU |

---

## ⚙️ Key Settings

| Setting | Default | Description |
|---|---|---|
| `MODEL_PATH` | `models/best.pt` | Path to YOLO model |
| `HEAD_CLASS` | `0` | Primary class ID for head/player |
| `HEAD_CLASSES` | `[0, 1]` | All class IDs to target |
| `CAPTURE_MODE` | `opencv` | Capture method (`opencv`, `ndi`, `mss`) |
| `IN_GAME_SENS` | `1.25` | In-game sensitivity |
| `DEADZONE_PX` | `2.5` | Deadzone radius in pixels |
| `MIN_CONFIDENCE` | `0.3` | Minimum detection confidence |
| `SMOOTHING_ALPHA` | `0.6` | Movement smoothing (0–1) |

---

## 🛠️ Capture Modes

| Mode | Latency | Setup | Use Case |
|---|---|---|---|
| **opencv** (DSHOW) | Low | None — reads capture card directly | Primary — capture card via DirectShow. No OBS needed. |
| **ndi** | Very Low | OBS + NDI plugin | Lowest latency NDI streaming (requires OBS) |
| **mss** | Medium | None | Single-PC fallback (⚠️ detectable) |

---

## 📁 Project Structure

```
YoloAimBot/
├── main.py                        # Main aimbot loop & threads
├── capture.py                     # Frame capture (OpenCV / NDI / MSS)
├── detection.py                   # YOLO model loading & inference
├── config.py                      # All settings
├── gui.py                         # Tkinter GUI
├── check_models.py                # Verify model class names
├── debug_viewer_capture_card.py   # Debug viewer with overlay
├── run.bat                        # Windows launcher
├── requirements.txt               # Python dependencies
└── models/                        # YOLO model files (not included)
    └── .gitkeep
```

---

## ❓ FAQ

**Q: Do I need OBS?**
A: **No.** The `opencv` mode reads the capture card directly via DirectShow — no OBS needed. OBS is only required for the `ndi` mode, or if you want to use Virtual Camera for overlays/filters.

**Q: Can I run this on a single PC without a capture card?**
A: The `mss` mode supports single-PC setups, but this is **highly detectable** by anti-cheat. A capture card + 2nd PC is the intended setup.

**Q: Does the 2nd PC need a GPU?**
A: No. YOLO runs on CPU acceptably for this use case. A modest laptop is sufficient.

**Q: Will this get me banned?**
A: Using any external aim-assist in online games carries a ban risk. This project is for offline/educational use only.

---

## ⚖️ Disclaimer

**This software is provided for educational purposes only.** I do not endorse cheating in online games. By using this software, you acknowledge that:

1. You are solely responsible for how you use it
2. Using this in online multiplayer games may violate the game's Terms of Service
3. You may face account bans or other penalties
4. This project is intended for learning about computer vision, real-time systems, and hardware integration

**Use at your own risk. Do not use in competitive or ranked game modes.**
