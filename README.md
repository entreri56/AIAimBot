# 🎯 YoloAimBot — Capture Card Edition

> **⚠️ EDUCATIONAL PURPOSES ONLY**
>
> This project is intended **solely for educational and research purposes** — to explore computer vision, real-time object detection with YOLO, and hardware-in-the-loop systems. **It is not intended for use in competitive online games.** Using this software in multiplayer games may violate the game's Terms of Service and result in permanent bans. The authors assume no liability for misuse.

---

## 📋 Overview

YoloAimBot is an AI-powered aim-assist tool built around **YOLO (You Only Look Once)** object detection. It runs on a **separate PC** (not the gaming PC) to avoid detection by anti-cheat systems.

```
┌─────────────┐    HDMI    ┌──────────────┐    Virtual Cam/NDI    ┌──────────────┐      USB      ┌─────────────┐
│  Gaming PC  │ ─────────→ │ Capture Card  │ ────────────────────→ │  This Bot    │ ────────────→ │   Gaming PC │
│  (CS2/etc)  │            │   + OBS PC    │                       │  (Python)    │    MAKCU      │  (mouse in) │
└─────────────┘            └──────────────┘                       └──────────────┘               └─────────────┘
```

1. **Gaming PC** outputs video over HDMI
2. **Capture Card** receives the feed on a 2nd PC running OBS
3. **OBS** outputs via Virtual Camera or NDI
4. **This program** captures frames, runs YOLO detection, and calculates mouse movements
5. **MAKCU** (Mouse And Keyboard Controlled via USB) sends movements back to the Gaming PC

---

## 🔌 Hardware Requirements

### **You MUST have a capture card — this is non-negotiable.**

| Component | Purpose |
|---|---|
| **Capture card** | Captures HDMI output from gaming PC (e.g., Elgato HD60 X, AverMedia Live Gamer, or generic USB 3.0 HDMI capture dongle) |
| **Second PC** | Runs OBS + this Python program — does NOT need a powerful GPU |
| **MAKCU device** | Hardware mouse emulator (Arduino/Teensy-based USB HID device) that connects via USB to the gaming PC |
| **HDMI cable(s)** | To connect the gaming PC to the capture card |

### Recommended capture cards:
- Elgato HD60 X / 4K X
- AverMedia Live Gamer Ultra
- Generic USB 3.0 HDMI Video Capture dongles ($15–$30 on Amazon/AliExpress)

### Why a capture card?
- Anti-cheat software cannot detect screen capture happening on a **different physical machine**
- OBS Virtual Camera presents the capture card feed as a webcam, which this program reads via OpenCV
---

## 🚀 Quick Start

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Get a YOLO model

Train your own model or download a pre-trained one. Place it in the `models/` folder:
```
models/
  └── best.pt        ← your trained YOLO model
```

For CS2, models are commonly trained on player/head datasets. Use `check_models.py` to verify:
```bash
python check_models.py
```

### 3. Set up OBS

1. Install [OBS Studio](https://obsproject.com/)
2. Add your capture card as a video source
3. Enable **OBS Virtual Camera** (Tools → Virtual Camera → Start)
4. (Optional) Install the [OBS-NDI plugin](https://github.com/obs-ndi/obs-ndi) for lower latency via NDI

### 4. Configure

Edit `config.py` or use the GUI:

```python
# config.py key settings:
CAPTURE_MODE  = "opencv"     # "opencv", "ndi", or "mss" (single-PC fallback)
CAMERA_INDEX  = 0            # OBS Virtual Camera device index
IN_GAME_SENS  = 1.25         # Match your in-game sensitivity
HEAD_CLASSES  = [0, 1]       # Target class IDs (CS2: 0=CT, 1=T)
```

### 5. Run

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
| **opencv** (DSHOW) | Low | OBS Virtual Camera | Primary — capture card via DirectShow |
| **ndi** | Very Low | OBS + NDI plugin | Lowest latency NDI streaming |
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

**Q: Can I run this on a single PC without a capture card?**
A: The `mss` mode supports single-PC setups, but this is **highly detectable** by anti-cheat. A capture card + 2nd PC is the intended setup.

**Q: Does the 2nd PC need a GPU?**
A: No. YOLO runs on CPU acceptably for this use case. A modest laptop is sufficient.

**Q: Will this get me banned?**
A: Using any external aim-assist in online games carries a ban risk. This project is for offline/educational use only.

---

## ⚖️ Disclaimer

**This software is provided for educational purposes only.** The authors do not endorse cheating in online games. By using this software, you acknowledge that:

1. You are solely responsible for how you use it
2. Using this in online multiplayer games may violate the game's Terms of Service
3. You may face account bans or other penalties
4. This project is intended for learning about computer vision, real-time systems, and hardware integration

**Use at your own risk. Do not use in competitive or ranked game modes.**