"""
Capture module for YoloAimBot - Capture Card Edition.
Supports three capture methods:
  1. opencv  - OBS Virtual Camera / capture card via DirectShow (cv2.VideoCapture)
  2. ndi     - NDI stream from OBS (requires cyndilib)
  3. mss     - Direct screen capture (single PC, fallback)
"""

import time
import threading
import numpy as np
import cv2
import queue

# ── OpenCV / Virtual Camera Capture ──────────────────────────────────
class OpenCVCamera:
    """
    Captures frames from a DirectShow device (OBS Virtual Camera, capture card, webcam).
    This is the primary method for capture card setups.
    """
    def __init__(self, camera_index=0, width=1920, height=1080, fps=60):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.running = False
        self._frame = None
        self._lock = threading.Lock()
        self._thread = None

    def start(self):
        """Open the camera and start the capture thread."""
        # Try DSHOW only (MSMF opens OBS VirtualCam but can't decode frames)
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            # Last resort: try default (but warn — may hit MSMF issues)
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_ANY)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {self.camera_index}. "
                f"Make sure OBS Virtual Camera is running or capture card is connected."
            )

        # Configure capture
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

        # Read actual resolution
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"[CAPTURE] Camera {self.camera_index}: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")

        # Test grab a frame to verify the backend actually works
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            self.cap.release()
            self.cap = None
            raise RuntimeError(
                f"Camera {self.camera_index} opened but cannot grab frames. "
                f"The backend can see the device but not decode its output. "
                f"Try a different camera index or restart OBS Virtual Camera."
            )

        # Store the test frame so get_latest_frame() returns something immediately
        with self._lock:
            self._frame = test_frame

        self.running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return actual_w, actual_h

    def _capture_loop(self):
        """Dedicated thread that continuously grabs frames for lowest latency."""
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.01)
                continue

            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.001)

    def get_latest_frame(self):
        """Get the most recent frame (non-blocking)."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def stop(self):
        """Stop capture and release resources."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        print("[CAPTURE] Camera released.")


# ── NDI Capture ──────────────────────────────────────────────────────
class NDICamera:
    """
    Captures frames from an NDI stream (OBS with NDI plugin).
    Requires: pip install cyndilib
    """
    def __init__(self, source_name=""):
        self.source_name = source_name
        self.receiver = None
        self.video_frame = None
        self.finder = None
        self.connected = False
        self._frame = None
        self._lock = threading.Lock()
        self._thread = None
        self.running = False
        self.width = 1920
        self.height = 1080

    def start(self):
        """Connect to NDI source and start capture."""
        try:
            from cyndilib.wrapper.ndi_recv import RecvColorFormat, RecvBandwidth
            from cyndilib.finder import Finder
            from cyndilib.receiver import Receiver
            from cyndilib.video_frame import VideoFrameSync
        except ImportError:
            raise ImportError(
                "cyndilib not installed. Install with: pip install cyndilib"
            )

        self.finder = Finder()
        self.finder.open()
        sources = self.finder.get_source_names() or []

        if not sources:
            self.finder.close()
            raise RuntimeError("No NDI sources found. Is OBS running with NDI output enabled?")

        print(f"[NDI] Available sources: {sources}")

        # Select source
        if self.source_name and self.source_name in sources:
            selected = self.source_name
        else:
            selected = sources[0]
            print(f"[NDI] Auto-selected source: {selected}")

        source = self.finder.get_source(selected)
        if not source:
            self.finder.close()
            raise RuntimeError(f"NDI source '{selected}' not available.")

        self.receiver = Receiver(
            color_format=RecvColorFormat.RGBX_RGBA,
            bandwidth=RecvBandwidth.highest,
        )
        self.video_frame = VideoFrameSync()
        self.receiver.frame_sync.set_video_frame(self.video_frame)
        self.receiver.set_source(source)

        # Wait for connection
        for _ in range(200):
            if self.receiver.is_connected():
                self.connected = True
                break
            time.sleep(0.01)

        if not self.connected:
            self.finder.close()
            raise RuntimeError("NDI connection timeout.")

        print(f"[NDI] Connected to: {selected}")

        self.running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self.width, self.height

    def _capture_loop(self):
        """Continuously grab NDI frames."""
        while self.running:
            if not self.receiver or not self.receiver.is_connected():
                time.sleep(0.01)
                continue

            try:
                self.receiver.frame_sync.capture_video()
                if min(self.video_frame.xres, self.video_frame.yres) == 0:
                    time.sleep(0.002)
                    continue

                self.width = self.video_frame.xres
                self.height = self.video_frame.yres

                frame = np.frombuffer(self.video_frame, dtype=np.uint8).copy()
                frame = frame.reshape((self.video_frame.yres, self.video_frame.xres, 4))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

                with self._lock:
                    self._frame = frame
            except Exception as e:
                print(f"[NDI] Frame error: {e}")
                time.sleep(0.005)

    def get_latest_frame(self):
        """Get the most recent NDI frame."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def stop(self):
        """Stop NDI capture."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        try:
            if self.receiver:
                self.receiver.set_source(None)
        except Exception:
            pass
        try:
            if self.finder:
                self.finder.close()
        except Exception:
            pass
        print("[NDI] Capture stopped.")


# ── MSS Screen Capture (Single PC Fallback) ──────────────────────────
class MSSCamera:
    """
    Direct screen capture using mss (for single-PC setups).
    """
    def __init__(self, monitor_index=1, capture_size=320, auto_center=True, center=(960, 540)):
        import mss
        self.sct = mss.mss()
        self.monitor_index = monitor_index
        self.capture_size = capture_size
        self.auto_center = auto_center
        self.center = center
        self.monitor = None
        self._setup_monitor()

    def _setup_monitor(self):
        """Configure the capture region."""
        mon = self.sct.monitors[self.monitor_index]
        if self.auto_center:
            cx = mon["left"] + mon["width"] // 2
            cy = mon["top"] + mon["height"] // 2
        else:
            cx, cy = self.center

        half = self.capture_size // 2
        self.monitor = {
            "left": cx - half,
            "top": cy - half,
            "width": self.capture_size,
            "height": self.capture_size,
        }
        print(f"[MSS] Capturing {self.capture_size}² px at ({cx}, {cy})")

    def get_latest_frame(self):
        """Capture a single frame."""
        img = np.array(self.sct.grab(self.monitor))
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def stop(self):
        """Release MSS resources."""
        self.sct.close()
        print("[MSS] Capture stopped.")


# ── Factory Function ─────────────────────────────────────────────────
def create_camera(config):
    """
    Create the appropriate camera based on config.CAPTURE_MODE.
    Returns (camera, width, height).
    """
    mode = config.CAPTURE_MODE.lower()

    if mode == "opencv":
        cam = OpenCVCamera(
            camera_index=config.CAMERA_INDEX,
            width=config.CAPTURE_WIDTH,
            height=config.CAPTURE_HEIGHT,
            fps=config.CAPTURE_FPS,
        )
        w, h = cam.start()
        return cam, w, h

    elif mode == "ndi":
        cam = NDICamera(source_name=config.NDI_SOURCE_NAME)
        w, h = cam.start()
        return cam, w, h

    elif mode == "mss":
        cam = MSSCamera(
            monitor_index=config.MONITOR_INDEX,
            capture_size=config.CAPTURE_SIZE,
            auto_center=config.AUTO_CENTER,
            center=config.CAPTURE_CENTER,
        )
        return cam, config.CAPTURE_SIZE, config.CAPTURE_SIZE

    else:
        raise ValueError(f"Unknown CAPTURE_MODE: {mode}")


# ── Utility: List available cameras ──────────────────────────────────
def list_cameras():
    """List all available DirectShow camera devices."""
    cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cameras.append((i, w, h))
            cap.release()
    return cameras


if __name__ == "__main__":
    # Test: list available cameras
    print("Available cameras:")
    for idx, w, h in list_cameras():
        print(f"  Camera {idx}: {w}x{h}")