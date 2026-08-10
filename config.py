"""
Configuration for YoloAimBot - Capture Card Edition
All settings in one place. Edit these values or use the GUI.
"""

import os

# ── MODEL ──────────────────────────────────────
MODEL_PATH           = "models/best.pt"     # Path to your YOLO model
HEAD_CLASS           = 0                    # Primary class ID for head/player to aim at
HEAD_CLASSES         = [0, 1]               # All class IDs to target (CS2: 0=CT, 1=T). Overrides HEAD_CLASS if set.
AIM_HEAD_Y_FRACTION  = 0.15                 # 0.0 = top of box, 0.5 = centre, 0.25 = forehead

# ── CAPTURE ────────────────────────────────────
# "opencv" = OBS Virtual Camera / capture card (via DirectShow)
# "ndi"    = NDI stream from OBS (requires NDI plugin)
# "mss"    = Direct screen capture (single PC only)
CAPTURE_MODE         = "opencv"

# OpenCV / Virtual Camera settings
CAMERA_INDEX         = 0                    # Which camera device? (0, 1, 2...)
CAPTURE_WIDTH        = 1920                 # Capture resolution width
CAPTURE_HEIGHT       = 1080                 # Capture resolution height
CAPTURE_FPS          = 60                   # Target FPS for capture

# NDI settings (only used if CAPTURE_MODE = "ndi")
NDI_SOURCE_NAME      = ""                   # Leave empty to auto-detect first source

# MSS settings (only used if CAPTURE_MODE = "mss")
MONITOR_INDEX        = 1                    # Which monitor to capture
CAPTURE_SIZE         = 320                  # FOV size (square region)
AUTO_CENTER          = True                 # Auto-center on monitor
CAPTURE_CENTER       = (960, 540)           # Manual center if AUTO_CENTER=False

# ── IN-GAME SENSITIVITY ────────────────────────
IN_GAME_SENS         = 1.25                 # Your in-game sensitivity

# ── AIMING ─────────────────────────────────────
DEADZONE_PX          = 2.5                  # No movement within this radius
BASE_SPEED           = 0.50                 # Base movement speed (0.0-1.0)
FAR_DISTANCE         = 200.0                # Distance considered "far"
CLOSE_DISTANCE       = 30.0                 # Distance considered "close"
FAR_SPEED_MULT       = 1.0                  # Speed multiplier when far
CLOSE_SPEED_MULT     = 0.10                 # Speed multiplier when close
SMOOTHING_ALPHA      = 0.6                  # Exponential smoothing (0-1)

# ── TARGET LOCK ────────────────────────────────
MIN_CONFIDENCE       = 0.3                  # Ignore detections below this
MAX_TARGET_DISTANCE  = 100.0                # Only consider targets within this many pixels of crosshair
LOCK_HYSTERESIS      = 60.0                 # New target must be this much closer
LOCK_LOST_FRAMES     = 8                    # Frames before giving up on lost target

# ── RATES ──────────────────────────────────────
DETECTION_DELAY      = 0.025                # Detection loop interval
MOVE_TICK            = 0.008                # Movement thread tick interval

# ── DEBUG ──────────────────────────────────────
DEBUG_PRINT          = False                # Verbose console output
SHOW_DEBUG_WINDOW    = False                # Show detection overlay window

# ── MAKCU ──────────────────────────────────────
# These are handled by the makcu library automatically
MAKCU_AUTO_RECONNECT = True
MAKCU_DEBUG          = False


def get_sensitivity_scale():
    """Convert in-game sensitivity to pixel-to-mouse-unit scale."""
    return 1.07437623 * pow(IN_GAME_SENS, -0.9936827126)


def print_settings():
    """Print current settings to console."""
    print(f"""
╔══════════════════════════════════════════════════════╗
║           YoloAimBot - Capture Card Edition          ║
╠══════════════════════════════════════════════════════╣
║  Model:       {MODEL_PATH:<38} ║
║  Head Class:  {HEAD_CLASS:<38} ║
║  Capture:     {CAPTURE_MODE:<38} ║
║  In-Game Sens:{IN_GAME_SENS:<38} ║
║  Deadzone:    {DEADZONE_PX} px{' ' * 34} ║
║  Base Speed:  {BASE_SPEED:<38} ║
║  Smoothing:   {SMOOTHING_ALPHA:<38} ║
║  Min Conf:    {MIN_CONFIDENCE:<38} ║
║  Debug:       {DEBUG_PRINT:<38} ║
╚══════════════════════════════════════════════════════╝
""")