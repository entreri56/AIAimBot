"""
YoloAimBot - Capture Card Edition
Main aimbot loop. Ties together capture, detection, and mouse control.

Architecture:
  Gaming PC → Capture Card → 2nd PC (OBS) → Virtual Camera/NDI → This Program → MAKCU → Gaming PC

Threads:
  1. Detection Thread: Capture frame → YOLO inference → Find best target → Update shared state
  2. Movement Thread: Read shared state → Smooth movement → Send to MAKCU
"""

import time
import threading
import math
import cv2
import numpy as np

from capture import create_camera, list_cameras
from detection import load_model, detect, extract_targets, draw_debug_overlay
import config as cfg


# ── Global State ─────────────────────────────────────────────────────
_aimbot_running = False
_aim_toggled_on = False
_aim_toggle_lock = threading.Lock()

# Target state (written by detection thread, read by movement thread)
_target_offset = (0.0, 0.0)     # Scaled (dx, dy) in mouse units
_target_lock = threading.Lock()
_has_target = False

# Target lock state (prevents switching between targets)
_locked_target_center = None    # (cx, cy) in capture-space pixels
_locked_target_lost = 0         # Consecutive frames without locked target
_locked_target_dist = 0.0       # Distance of locked target from crosshair
_lock_lock = threading.Lock()

# Smoothed target position (in screen pixels, smoothed toward chosen target)
_smooth_target_x = 0.0
_smooth_target_y = 0.0
_smooth_target_initialized = False
_smooth_lock = threading.Lock()
_carry_x = 0.0
_carry_y = 0.0

# FPS tracking
_fps = 0.0
_fps_lock = threading.Lock()

# Detection count tracking (for GUI display)
_detection_count = 0
_detection_info = ""  # e.g. "2 targets | best: conf=0.82 dist=45px"
_detection_lock = threading.Lock()

# Debug frame (for GUI display)
_debug_frame = None
_debug_lock = threading.Lock()

# Debug display thread
_debug_display_thread = None
_debug_display_running = False


# ── Sensitivity Scaling ──────────────────────────────────────────────
def get_sensitivity_scale():
    """Convert in-game sensitivity to pixel-to-mouse-unit scale."""
    return 1.07437623 * math.pow(cfg.IN_GAME_SENS, -0.9936827126)


# ── Movement Thread ──────────────────────────────────────────────────
def movement_loop(controller, capture_width, capture_height):
    """
    Dedicated movement thread.
    Smooths the target position in screen space, then converts to mouse movement.
    This prevents the crosshair from jumping when the detection switches targets.
    """
    global _smooth_target_x, _smooth_target_y, _smooth_target_initialized
    global _carry_x, _carry_y

    crosshair_x = capture_width // 2
    crosshair_y = capture_height // 2
    sens_scale = get_sensitivity_scale()

    print("[MOVEMENT] Thread started")
    while _aimbot_running:
        with _aim_toggle_lock:
            active = _aim_toggled_on

        if not active:
            _smooth_target_initialized = False
            _carry_x = 0.0
            _carry_y = 0.0
            time.sleep(0.005)
            continue

        # Read latest target (raw offset from detection thread)
        with _target_lock:
            target_dx, target_dy = _target_offset
            has_target = _has_target

        if not has_target:
            # No target — decay smoothed position toward crosshair center
            if _smooth_target_initialized:
                _smooth_target_x += (crosshair_x - _smooth_target_x) * 0.3
                _smooth_target_y += (crosshair_y - _smooth_target_y) * 0.3
            _carry_x = 0.0
            _carry_y = 0.0
            time.sleep(cfg.MOVE_TICK)
            continue

        # Convert raw offset to absolute screen position of the target
        target_screen_x = crosshair_x + target_dx / sens_scale
        target_screen_y = crosshair_y + target_dy / sens_scale

        # Initialize or smooth toward the target position
        if not _smooth_target_initialized:
            _smooth_target_x = target_screen_x
            _smooth_target_y = target_screen_y
            _smooth_target_initialized = True
        else:
            # Exponential smoothing on screen position (not offset)
            # This means when the detection switches targets, the aim point
            # smoothly slides from the old position to the new one
            _smooth_target_x += (target_screen_x - _smooth_target_x) * cfg.SMOOTHING_ALPHA
            _smooth_target_y += (target_screen_y - _smooth_target_y) * cfg.SMOOTHING_ALPHA

        # Convert smoothed screen position back to offset from crosshair
        smooth_dx = (_smooth_target_x - crosshair_x) * sens_scale
        smooth_dy = (_smooth_target_y - crosshair_y) * sens_scale
        dist = math.hypot(smooth_dx, smooth_dy)

        # Deadzone check
        if dist < cfg.DEADZONE_PX:
            _carry_x = 0.0
            _carry_y = 0.0
            time.sleep(cfg.MOVE_TICK)
            continue

        # Dynamic speed scaling
        if dist > cfg.FAR_DISTANCE:
            speed_mult = cfg.FAR_SPEED_MULT
        elif dist < cfg.CLOSE_DISTANCE:
            speed_mult = cfg.CLOSE_SPEED_MULT
        else:
            ratio = (dist - cfg.CLOSE_DISTANCE) / (cfg.FAR_DISTANCE - cfg.CLOSE_DISTANCE)
            speed_mult = cfg.CLOSE_SPEED_MULT + ratio * (cfg.FAR_SPEED_MULT - cfg.CLOSE_SPEED_MULT)

        # Calculate movement with sub-pixel precision
        move_x = smooth_dx * cfg.BASE_SPEED * speed_mult
        move_y = smooth_dy * cfg.BASE_SPEED * speed_mult

        # Accumulate fractional parts
        _carry_x += move_x
        _carry_y += move_y

        int_x = int(_carry_x)
        int_y = int(_carry_y)

        _carry_x -= int_x
        _carry_y -= int_y

        if int_x != 0 or int_y != 0:
            controller.move(int_x, int_y)
            if cfg.DEBUG_PRINT:
                print(f"[MOVE] ({int_x}, {int_y}) | smooth=({smooth_dx:.1f},{smooth_dy:.1f}) "
                      f"| dist={dist:.1f} | speed={speed_mult:.2f}")

        time.sleep(cfg.MOVE_TICK)

    print("[MOVEMENT] Thread stopped")


# ── Detection Thread ─────────────────────────────────────────────────
def detection_loop(camera, capture_width, capture_height):
    """
    Main detection loop.
    Captures frames, runs YOLO, finds best target, updates shared state.
    """
    global _has_target, _target_offset, _locked_target_center, _locked_target_lost
    global _fps, _debug_frame

    # Load model
    model, class_names = load_model(cfg.MODEL_PATH)
    head_class_name = class_names.get(cfg.HEAD_CLASS, f"class_{cfg.HEAD_CLASS}")

    sens_scale = get_sensitivity_scale()
    crosshair_x = capture_width // 2
    crosshair_y = capture_height // 2

    frame_count = 0
    fps_start = time.perf_counter()

    print(f"[DETECTION] Thread started. Resolution: {capture_width}x{capture_height}")
    print(f"[DETECTION] Head class: {cfg.HEAD_CLASS} -> {head_class_name}")

    while _aimbot_running:
        with _aim_toggle_lock:
            active = _aim_toggled_on

        if not active:
            with _target_lock:
                _has_target = False
                _target_offset = (0.0, 0.0)
            time.sleep(0.01)
            continue

        # Capture frame
        frame = camera.get_latest_frame()
        if frame is None:
            time.sleep(0.001)
            continue

        # Run YOLO detection
        results = detect(
            model, frame,
            conf=cfg.MIN_CONFIDENCE,
            verbose=False,
        )

        # Extract targets (supports multiple head classes for CS2 CT+T models)
        targets = extract_targets(
            results, class_names,
            head_class=cfg.HEAD_CLASS,
            min_conf=cfg.MIN_CONFIDENCE,
            aim_y_fraction=cfg.AIM_HEAD_Y_FRACTION,
            head_classes=cfg.HEAD_CLASSES if hasattr(cfg, 'HEAD_CLASSES') and cfg.HEAD_CLASSES else None,
        )

        # Calculate distances from crosshair
        for t in targets:
            t["dist"] = math.hypot(t["x"] - crosshair_x, t["y"] - crosshair_y)

        # Filter: only keep targets within MAX_TARGET_DISTANCE of crosshair
        targets = [t for t in targets if t["dist"] <= cfg.MAX_TARGET_DISTANCE]

        found_target = False
        new_dx, new_dy = 0.0, 0.0
        chosen = None

        if targets:
            # ── Target Lock Logic ──
            with _lock_lock:
                locked = _locked_target_center
                lost_frames = _locked_target_lost

            chosen = None

            if locked is not None:
                # Try to find the locked target in current detections
                best_match = None
                best_match_dist = float('inf')
                for t in targets:
                    match_dist = math.hypot(t["cx"] - locked[0], t["cy"] - locked[1])
                    if match_dist < 80:  # Within 80px of last known position
                        if match_dist < best_match_dist:
                            best_match = t
                            best_match_dist = match_dist

                if best_match is not None:
                    # Locked target still visible — keep it
                    chosen = best_match
                    with _lock_lock:
                        _locked_target_lost = 0
                        _locked_target_center = (best_match["cx"], best_match["cy"])
                        _locked_target_dist = best_match["dist"]
                else:
                    # Locked target not found this frame
                    with _lock_lock:
                        _locked_target_lost = lost_frames + 1

                    if _locked_target_lost >= cfg.LOCK_LOST_FRAMES:
                        # Give up, pick closest
                        with _lock_lock:
                            _locked_target_center = None
                            _locked_target_lost = 0
                            _locked_target_dist = 0.0
                        targets.sort(key=lambda t: t["dist"])
                        chosen = targets[0]
                        with _lock_lock:
                            _locked_target_center = (chosen["cx"], chosen["cy"])
                            _locked_target_dist = chosen["dist"]
            else:
                # No current lock — pick closest target
                targets.sort(key=lambda t: t["dist"])
                chosen = targets[0]
                with _lock_lock:
                    _locked_target_center = (chosen["cx"], chosen["cy"])
                    _locked_target_lost = 0
                    _locked_target_dist = chosen["dist"]

            # ── Hysteresis check: don't switch to a new target unless it's
            #     significantly closer than the current locked one ──
            if chosen is not None and locked is not None:
                # If the chosen target is different from the locked one
                # (more than 30px away from locked position), check hysteresis
                chosen_dist_from_locked = math.hypot(
                    chosen["cx"] - locked[0], chosen["cy"] - locked[1]
                )
                if chosen_dist_from_locked > 30:
                    # This is a different target — only switch if it's
                    # LOCK_HYSTERESIS pixels closer to crosshair
                    with _lock_lock:
                        current_locked_dist = _locked_target_dist
                    if chosen["dist"] > current_locked_dist - cfg.LOCK_HYSTERESIS:
                        # Not enough improvement — stick with locked target
                        # Find the locked target in current detections
                        for t in targets:
                            t_dist = math.hypot(t["cx"] - locked[0], t["cy"] - locked[1])
                            if t_dist < 30:
                                chosen = t
                                break
                        else:
                            # Locked target not in current frame, but we haven't
                            # exceeded LOCK_LOST_FRAMES yet — keep waiting
                            chosen = None

            if chosen is not None:
                raw_dx = chosen["x"] - crosshair_x
                raw_dy = chosen["y"] - crosshair_y
                new_dx = raw_dx * sens_scale
                new_dy = raw_dy * sens_scale
                found_target = True

                if cfg.DEBUG_PRINT:
                    lock_status = "LOCKED" if (_locked_target_center is not None) else "FREE"
                    print(f"[DETECT] offset ({raw_dx:+.1f}, {raw_dy:+.1f}) px "
                          f"-> scaled ({new_dx:.1f}, {new_dy:.1f}) | "
                          f"dist={chosen['dist']:.1f} | conf={chosen['conf']:.2f} | {lock_status}")

        # Update shared state
        with _target_lock:
            _has_target = found_target
            _target_offset = (new_dx, new_dy)

        if not found_target:
            with _lock_lock:
                if _locked_target_center is not None:
                    _locked_target_lost += 1
                    if _locked_target_lost >= cfg.LOCK_LOST_FRAMES:
                        _locked_target_center = None
                        _locked_target_lost = 0

        # Debug window
        if cfg.SHOW_DEBUG_WINDOW:
            debug_img = draw_debug_overlay(
                frame, targets,
                locked_target=_locked_target_center,
                crosshair=(crosshair_x, crosshair_y),
                head_class_name=head_class_name,
            )
            # Also overlay FPS
            cv2.putText(debug_img, f"FPS: {_fps:.1f}", (debug_img.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            with _debug_lock:
                _debug_frame = debug_img

        # Update detection count for GUI
        with _detection_lock:
            _detection_count = len(targets)
            if chosen is not None and found_target:
                _detection_info = (f"{len(targets)} targets | "
                                   f"conf={chosen['conf']:.2f} "
                                   f"dist={chosen['dist']:.0f}px")
            elif len(targets) > 0:
                _detection_info = f"{len(targets)} targets | no lock"
            else:
                _detection_info = "0 targets"

        # FPS calculation
        frame_count += 1
        elapsed = time.perf_counter() - fps_start
        if elapsed > 1.0:
            with _fps_lock:
                _fps = frame_count / elapsed
            fps_start = time.perf_counter()
            frame_count = 0

        time.sleep(cfg.DETECTION_DELAY)

    print("[DETECTION] Thread stopped")


# ── Public API ────────────────────────────────────────────────────────
def get_fps():
    """Get current detection FPS."""
    with _fps_lock:
        return _fps


def get_detection_count():
    """Get current detection count for GUI display."""
    with _detection_lock:
        return _detection_count


def get_detection_info():
    """Get formatted detection info string for GUI display."""
    with _detection_lock:
        return _detection_info


def get_debug_frame():
    """Get the latest debug frame for GUI display."""
    with _debug_lock:
        if _debug_frame is None:
            return None
        return _debug_frame.copy()


def is_aimbot_running():
    """Check if aimbot is running."""
    return _aimbot_running


def is_aim_toggled():
    """Check if aiming is toggled on."""
    with _aim_toggle_lock:
        return _aim_toggled_on


def toggle_aim():
    """Toggle aim on/off."""
    global _aim_toggled_on
    with _aim_toggle_lock:
        _aim_toggled_on = not _aim_toggled_on
    state = "ON" if _aim_toggled_on else "OFF"
    print(f"[TOGGLE] Aim {state}")
    return _aim_toggled_on


def set_aim_state(state: bool):
    """Set aim state directly."""
    global _aim_toggled_on
    with _aim_toggle_lock:
        _aim_toggled_on = state


def start_aimbot():
    """
    Start the aimbot. Initializes camera, model, and starts threads.
    Returns the controller object for external use.
    """
    global _aimbot_running, _debug_display_running

    # Reset in case of stale state from previous crash
    _aimbot_running = False
    _debug_display_running = False

    _aimbot_running = True

    # Print settings
    cfg.print_settings()

    # List available cameras
    print("\n[INFO] Available cameras:")
    for idx, w, h in list_cameras():
        print(f"  Camera {idx}: {w}x{h}")

    # Create camera
    print(f"\n[INFO] Initializing capture ({cfg.CAPTURE_MODE})...")
    camera, cap_w, cap_h = create_camera(cfg)

    # Connect to MAKCU
    print("[MAKCU] Connecting...")
    from makcu import create_controller, MouseButton
    controller = create_controller(debug=cfg.MAKCU_DEBUG, auto_reconnect=cfg.MAKCU_AUTO_RECONNECT)

    # Set up button callback for toggle
    def on_button_event(button: MouseButton, pressed: bool):
        if button == MouseButton.RIGHT and pressed:
            toggle_aim()

    controller.set_button_callback(on_button_event)
    controller.enable_button_monitoring(True)
    print("[MAKCU] Connected!")

    # Start threads
    threading.Thread(target=movement_loop, args=(controller, cap_w, cap_h), daemon=True).start()
    threading.Thread(target=detection_loop, args=(camera, cap_w, cap_h), daemon=True).start()

    print("\n[INFO] Aimbot started! Press RIGHT mouse button to toggle aim.")
    print("[INFO] Press Ctrl+C to exit.\n")

    return controller


def stop_aimbot(camera=None, controller=None):
    """Stop the aimbot and clean up resources."""
    global _aimbot_running, _debug_display_running
    _aimbot_running = False
    _debug_display_running = False

    if camera:
        try:
            camera.stop()
        except Exception as e:
            print(f"[WARN] Camera cleanup error: {e}")

    if controller:
        try:
            controller.disconnect()
        except Exception as e:
            print(f"[WARN] Controller cleanup error: {e}")

    print("[INFO] Aimbot stopped.")


# ── Standalone Entry Point ───────────────────────────────────────────
if __name__ == "__main__":
    controller = start_aimbot()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
        stop_aimbot()
        print("Bye!")