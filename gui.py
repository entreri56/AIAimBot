"""
GUI for YoloAimBot - Capture Card Edition.
Simple, clean interface using tkinter (no external GUI library needed).
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
import sys

# Add parent directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from capture import list_cameras


class AimbotGUI:
    """Main GUI window for the aimbot."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("YoloAimBot - Capture Card Edition")
        self.root.geometry("700x750")
        self.root.configure(bg="#1a1a1a")
        self.root.resizable(True, True)
        self.root.minsize(600, 650)

        # State
        self.controller = None
        self.camera = None
        self.running = False
        self.aim_active = False

        # Style
        self._setup_style()

        # Build UI
        self._build_ui()

        # Refresh camera list
        self._refresh_cameras()

        # Window close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_style(self):
        """Configure ttk styles for dark theme."""
        style = ttk.Style()
        style.theme_use("clam")

        # Colors
        BG = "#1a1a1a"
        FG = "#ffffff"
        ACCENT = "#ff1744"
        ACCENT_HOVER = "#d50000"
        ENTRY_BG = "#2a2a2a"
        FRAME_BG = "#222222"

        style.configure("TFrame", background=BG)
        style.configure("TLabelframe", background=BG, foreground=ACCENT, borderwidth=1)
        style.configure("TLabelframe.Label", background=BG, foreground=ACCENT, font=("Segoe UI", 11, "bold"))
        style.configure("TLabel", background=BG, foreground=FG, font=("Segoe UI", 10))
        style.configure("TButton", background=ACCENT, foreground=FG, font=("Segoe UI", 10, "bold"), borderwidth=0)
        style.map("TButton", background=[("active", ACCENT_HOVER)])
        style.configure("TEntry", fieldbackground=ENTRY_BG, foreground=FG, insertcolor=FG)
        style.configure("TSpinbox", fieldbackground=ENTRY_BG, foreground=FG)
        style.configure("TCheckbutton", background=BG, foreground=FG)
        style.configure("TCombobox", fieldbackground=ENTRY_BG, foreground=FG, background=ENTRY_BG)
        style.configure("TScale", background=BG)

        # Custom styles
        style.configure("Start.TButton", background="#00c853", foreground=FG, font=("Segoe UI", 12, "bold"))
        style.map("Start.TButton", background=[("active", "#00e676")])
        style.configure("Stop.TButton", background=ACCENT, foreground=FG, font=("Segoe UI", 12, "bold"))
        style.map("Stop.TButton", background=[("active", ACCENT_HOVER)])
        style.configure("Status.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground=ACCENT)

    def _build_ui(self):
        """Build the complete UI."""
        # ── Title ──
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill="x", padx=15, pady=(15, 5))

        ttk.Label(title_frame, text="YoloAimBot", style="Header.TLabel").pack(side="left")
        ttk.Label(title_frame, text="Capture Card Edition", 
                  font=("Segoe UI", 10), foreground="#888").pack(side="left", padx=10)

        # ── Status Bar ──
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill="x", padx=15, pady=5)

        self.status_label = ttk.Label(status_frame, text="● Stopped", style="Status.TLabel", foreground="#ff1744")
        self.status_label.pack(side="left")

        self.fps_label = ttk.Label(status_frame, text="FPS: --", foreground="#00e676")
        self.fps_label.pack(side="right")

        self.det_label = ttk.Label(status_frame, text="Detections: --", foreground="#888")
        self.det_label.pack(side="right", padx=20)

        self.aim_label = ttk.Label(status_frame, text="Aim: OFF", foreground="#888")
        self.aim_label.pack(side="right", padx=20)

        # ── Notebook (Tabs) ──
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=15, pady=10)

        # Tab 1: Capture
        capture_tab = ttk.Frame(notebook)
        notebook.add(capture_tab, text="  Capture  ")
        self._build_capture_tab(capture_tab)

        # Tab 2: Model & Detection
        model_tab = ttk.Frame(notebook)
        notebook.add(model_tab, text="  Model & Detection  ")
        self._build_model_tab(model_tab)

        # Tab 3: Aiming
        aim_tab = ttk.Frame(notebook)
        notebook.add(aim_tab, text="  Aiming  ")
        self._build_aim_tab(aim_tab)

        # Tab 4: Advanced
        adv_tab = ttk.Frame(notebook)
        notebook.add(adv_tab, text="  Advanced  ")
        self._build_advanced_tab(adv_tab)

        # ── Control Buttons ──
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", padx=15, pady=(5, 15))

        self.start_btn = ttk.Button(btn_frame, text="▶ START AIMBOT", style="Start.TButton",
                                     command=self._start_aimbot, width=20)
        self.start_btn.pack(side="left", padx=(0, 10))

        self.stop_btn = ttk.Button(btn_frame, text="⏹ STOP", style="Stop.TButton",
                                    command=self._stop_aimbot, width=15, state="disabled")
        self.stop_btn.pack(side="left", padx=(0, 10))

        self.toggle_btn = ttk.Button(btn_frame, text="Toggle Aim (Right Click)", 
                                      command=self._toggle_aim, width=25)
        self.toggle_btn.pack(side="left")

        # ── Log ──
        log_frame = ttk.LabelFrame(self.root, text="Log")
        log_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        self.log_text = tk.Text(log_frame, height=6, bg="#0a0a0a", fg="#00ff00", 
                                font=("Consolas", 9), wrap="word", state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

        scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _build_capture_tab(self, parent):
        """Build the capture settings tab."""
        frame = ttk.Frame(parent, padding=15)
        frame.pack(fill="both", expand=True)

        # Capture mode
        ttk.Label(frame, text="Capture Mode:").grid(row=0, column=0, sticky="w", pady=5)
        self.capture_mode_var = tk.StringVar(value=cfg.CAPTURE_MODE)
        mode_combo = ttk.Combobox(frame, textvariable=self.capture_mode_var, 
                                   values=["opencv", "ndi", "mss"], state="readonly", width=15)
        mode_combo.grid(row=0, column=1, sticky="w", pady=5, padx=10)
        mode_combo.bind("<<ComboboxSelected>>", self._on_capture_mode_change)

        # Camera selection (OpenCV mode)
        self.camera_frame = ttk.LabelFrame(frame, text="Camera Settings (OpenCV)")
        self.camera_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=10)

        ttk.Label(self.camera_frame, text="Camera:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.camera_var = tk.StringVar(value=str(cfg.CAMERA_INDEX))
        self.camera_combo = ttk.Combobox(self.camera_frame, textvariable=self.camera_var, 
                                          values=["0"], state="readonly", width=20)
        self.camera_combo.grid(row=0, column=1, sticky="w", padx=10, pady=5)
        ttk.Button(self.camera_frame, text="Refresh", command=self._refresh_cameras, width=10).grid(
            row=0, column=2, padx=10, pady=5)

        ttk.Label(self.camera_frame, text="Resolution:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        res_frame = ttk.Frame(self.camera_frame)
        res_frame.grid(row=1, column=1, columnspan=2, sticky="w", padx=10, pady=5)
        self.cap_width_var = tk.StringVar(value=str(cfg.CAPTURE_WIDTH))
        self.cap_height_var = tk.StringVar(value=str(cfg.CAPTURE_HEIGHT))
        ttk.Entry(res_frame, textvariable=self.cap_width_var, width=6).pack(side="left")
        ttk.Label(res_frame, text=" × ").pack(side="left")
        ttk.Entry(res_frame, textvariable=self.cap_height_var, width=6).pack(side="left")

        ttk.Label(self.camera_frame, text="Target FPS:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.cap_fps_var = tk.StringVar(value=str(cfg.CAPTURE_FPS))
        ttk.Spinbox(self.camera_frame, textvariable=self.cap_fps_var, from_=15, to=240, 
                     width=8).grid(row=2, column=1, sticky="w", padx=10, pady=5)

        # NDI settings
        self.ndi_frame = ttk.LabelFrame(frame, text="NDI Settings")
        self.ndi_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=10)
        ttk.Label(self.ndi_frame, text="Source Name:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.ndi_source_var = tk.StringVar(value=cfg.NDI_SOURCE_NAME)
        ttk.Entry(self.ndi_frame, textvariable=self.ndi_source_var, width=30).grid(
            row=0, column=1, sticky="w", padx=10, pady=5)
        ttk.Label(self.ndi_frame, text="(leave empty for auto-detect)", 
                  foreground="#888", font=("Segoe UI", 8)).grid(row=1, column=1, sticky="w", padx=10)

        # MSS settings
        self.mss_frame = ttk.LabelFrame(frame, text="MSS Settings (Single PC)")
        self.mss_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=10)
        ttk.Label(self.mss_frame, text="Monitor:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.monitor_var = tk.StringVar(value=str(cfg.MONITOR_INDEX))
        ttk.Spinbox(self.mss_frame, textvariable=self.monitor_var, from_=0, to=10, 
                     width=5).grid(row=0, column=1, sticky="w", padx=10, pady=5)
        ttk.Label(self.mss_frame, text="FOV Size:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.fov_var = tk.StringVar(value=str(cfg.CAPTURE_SIZE))
        ttk.Spinbox(self.mss_frame, textvariable=self.fov_var, from_=100, to=800, 
                     width=8).grid(row=1, column=1, sticky="w", padx=10, pady=5)

        # Update visibility
        self._on_capture_mode_change()

    def _build_model_tab(self, parent):
        """Build the model & detection settings tab."""
        frame = ttk.Frame(parent, padding=15)
        frame.pack(fill="both", expand=True)

        # Model path
        ttk.Label(frame, text="Model Path:").grid(row=0, column=0, sticky="w", pady=5)
        model_frame = ttk.Frame(frame)
        model_frame.grid(row=0, column=1, sticky="ew", pady=5, padx=10)
        self.model_path_var = tk.StringVar(value=cfg.MODEL_PATH)
        ttk.Entry(model_frame, textvariable=self.model_path_var, width=35).pack(side="left")
        ttk.Button(model_frame, text="Browse", command=self._browse_model, width=8).pack(side="left", padx=5)

        # Head class
        ttk.Label(frame, text="Head Class ID:").grid(row=1, column=0, sticky="w", pady=5)
        self.head_class_var = tk.StringVar(value=str(cfg.HEAD_CLASS))
        ttk.Spinbox(frame, textvariable=self.head_class_var, from_=0, to=99, width=8).grid(
            row=1, column=1, sticky="w", pady=5, padx=10)

        # Aim Y fraction
        ttk.Label(frame, text="Aim Y Fraction:").grid(row=2, column=0, sticky="w", pady=5)
        self.aim_y_var = tk.StringVar(value=str(cfg.AIM_HEAD_Y_FRACTION))
        ttk.Scale(frame, from_=0.0, to=0.5, variable=self.aim_y_var, orient="horizontal", 
                  length=200).grid(row=2, column=1, sticky="w", pady=5, padx=10)
        ttk.Label(frame, text="(0=top of head, 0.5=center)").grid(row=2, column=2, sticky="w", pady=5)

        # Min confidence
        ttk.Label(frame, text="Min Confidence:").grid(row=3, column=0, sticky="w", pady=5)
        self.min_conf_var = tk.StringVar(value=str(cfg.MIN_CONFIDENCE))
        ttk.Scale(frame, from_=0.1, to=0.9, variable=self.min_conf_var, orient="horizontal", 
                  length=200).grid(row=3, column=1, sticky="w", pady=5, padx=10)

        # Debug
        self.debug_var = tk.BooleanVar(value=cfg.DEBUG_PRINT)
        ttk.Checkbutton(frame, text="Verbose Debug Output", variable=self.debug_var).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=10)

        self.show_debug_var = tk.BooleanVar(value=cfg.SHOW_DEBUG_WINDOW)
        ttk.Checkbutton(frame, text="Show Debug Window (detection overlay)", 
                        variable=self.show_debug_var).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=5)

    def _build_aim_tab(self, parent):
        """Build the aiming settings tab."""
        frame = ttk.Frame(parent, padding=15)
        frame.pack(fill="both", expand=True)

        # In-game sensitivity
        ttk.Label(frame, text="In-Game Sensitivity:").grid(row=0, column=0, sticky="w", pady=5)
        self.sens_var = tk.StringVar(value=str(cfg.IN_GAME_SENS))
        ttk.Scale(frame, from_=0.1, to=10.0, variable=self.sens_var, orient="horizontal", 
                  length=200).grid(row=0, column=1, sticky="w", pady=5, padx=10)

        # Base speed
        ttk.Label(frame, text="Base Speed:").grid(row=1, column=0, sticky="w", pady=5)
        self.base_speed_var = tk.StringVar(value=str(cfg.BASE_SPEED))
        ttk.Scale(frame, from_=0.05, to=1.0, variable=self.base_speed_var, orient="horizontal", 
                  length=200).grid(row=1, column=1, sticky="w", pady=5, padx=10)

        # Deadzone
        ttk.Label(frame, text="Deadzone (px):").grid(row=2, column=0, sticky="w", pady=5)
        self.deadzone_var = tk.StringVar(value=str(cfg.DEADZONE_PX))
        ttk.Scale(frame, from_=0.0, to=10.0, variable=self.deadzone_var, orient="horizontal", 
                  length=200).grid(row=2, column=1, sticky="w", pady=5, padx=10)

        # Smoothing
        ttk.Label(frame, text="Smoothing Alpha:").grid(row=3, column=0, sticky="w", pady=5)
        self.smooth_var = tk.StringVar(value=str(cfg.SMOOTHING_ALPHA))
        ttk.Scale(frame, from_=0.1, to=1.0, variable=self.smooth_var, orient="horizontal", 
                  length=200).grid(row=3, column=1, sticky="w", pady=5, padx=10)

        # Speed multipliers
        ttk.Separator(frame, orient="horizontal").grid(row=4, column=0, columnspan=2, 
                                                         sticky="ew", pady=10)

        ttk.Label(frame, text="Close Speed Mult:").grid(row=5, column=0, sticky="w", pady=5)
        self.close_speed_var = tk.StringVar(value=str(cfg.CLOSE_SPEED_MULT))
        ttk.Scale(frame, from_=0.01, to=0.5, variable=self.close_speed_var, orient="horizontal", 
                  length=200).grid(row=5, column=1, sticky="w", pady=5, padx=10)

        ttk.Label(frame, text="Far Speed Mult:").grid(row=6, column=0, sticky="w", pady=5)
        self.far_speed_var = tk.StringVar(value=str(cfg.FAR_SPEED_MULT))
        ttk.Scale(frame, from_=0.5, to=2.0, variable=self.far_speed_var, orient="horizontal", 
                  length=200).grid(row=6, column=1, sticky="w", pady=5, padx=10)

        ttk.Label(frame, text="Close Distance:").grid(row=7, column=0, sticky="w", pady=5)
        self.close_dist_var = tk.StringVar(value=str(cfg.CLOSE_DISTANCE))
        ttk.Scale(frame, from_=5, to=100, variable=self.close_dist_var, orient="horizontal", 
                  length=200).grid(row=7, column=1, sticky="w", pady=5, padx=10)

        ttk.Label(frame, text="Far Distance:").grid(row=8, column=0, sticky="w", pady=5)
        self.far_dist_var = tk.StringVar(value=str(cfg.FAR_DISTANCE))
        ttk.Scale(frame, from_=50, to=500, variable=self.far_dist_var, orient="horizontal", 
                  length=200).grid(row=8, column=1, sticky="w", pady=5, padx=10)

    def _build_advanced_tab(self, parent):
        """Build the advanced settings tab."""
        frame = ttk.Frame(parent, padding=15)
        frame.pack(fill="both", expand=True)

        # Target lock
        ttk.Label(frame, text="Lock Lost Frames:").grid(row=0, column=0, sticky="w", pady=5)
        self.lock_frames_var = tk.StringVar(value=str(cfg.LOCK_LOST_FRAMES))
        ttk.Spinbox(frame, textvariable=self.lock_frames_var, from_=1, to=60, width=8).grid(
            row=0, column=1, sticky="w", pady=5, padx=10)

        # Detection delay
        ttk.Label(frame, text="Detection Delay (s):").grid(row=1, column=0, sticky="w", pady=5)
        self.det_delay_var = tk.StringVar(value=str(cfg.DETECTION_DELAY))
        ttk.Spinbox(frame, textvariable=self.det_delay_var, from_=0.005, to=0.2, 
                     increment=0.005, width=8).grid(row=1, column=1, sticky="w", pady=5, padx=10)

        # Move tick
        ttk.Label(frame, text="Move Tick (s):").grid(row=2, column=0, sticky="w", pady=5)
        self.move_tick_var = tk.StringVar(value=str(cfg.MOVE_TICK))
        ttk.Spinbox(frame, textvariable=self.move_tick_var, from_=0.001, to=0.05, 
                     increment=0.001, width=8).grid(row=2, column=1, sticky="w", pady=5, padx=10)

        # MAKCU settings
        ttk.Separator(frame, orient="horizontal").grid(row=3, column=0, columnspan=2, 
                                                         sticky="ew", pady=10)
        ttk.Label(frame, text="MAKCU Settings", font=("Segoe UI", 11, "bold"), 
                  foreground="#ff1744").grid(row=4, column=0, columnspan=2, sticky="w", pady=5)

        self.makcu_auto_var = tk.BooleanVar(value=cfg.MAKCU_AUTO_RECONNECT)
        ttk.Checkbutton(frame, text="Auto Reconnect", variable=self.makcu_auto_var).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=5)

        self.makcu_debug_var = tk.BooleanVar(value=cfg.MAKCU_DEBUG)
        ttk.Checkbutton(frame, text="MAKCU Debug Mode", variable=self.makcu_debug_var).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=5)

        # Save/Load
        ttk.Separator(frame, orient="horizontal").grid(row=7, column=0, columnspan=2, 
                                                         sticky="ew", pady=10)
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=8, column=0, columnspan=2, sticky="w", pady=5)
        ttk.Button(btn_frame, text="Save Settings", command=self._save_settings).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Load Settings", command=self._load_settings).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Reset Defaults", command=self._reset_settings).pack(side="left", padx=5)

    # ── Callbacks ────────────────────────────────────────────────────
    def _on_capture_mode_change(self, event=None):
        """Show/hide capture settings based on mode."""
        mode = self.capture_mode_var.get()
        if mode == "opencv":
            self.camera_frame.grid()
            self.ndi_frame.grid_remove()
            self.mss_frame.grid_remove()
        elif mode == "ndi":
            self.camera_frame.grid_remove()
            self.ndi_frame.grid()
            self.mss_frame.grid_remove()
        elif mode == "mss":
            self.camera_frame.grid_remove()
            self.ndi_frame.grid_remove()
            self.mss_frame.grid()

    def _refresh_cameras(self):
        """Refresh the camera list."""
        cameras = list_cameras()
        values = [f"{idx} ({w}x{h})" for idx, w, h in cameras]
        if not values:
            values = ["0 (no cameras found)"]
        self.camera_combo["values"] = values
        if values:
            self.camera_combo.current(0)

    def _browse_model(self):
        """Open file dialog to select model."""
        path = filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[("Model files", "*.pt *.onnx *.engine"), ("All files", "*.*")]
        )
        if path:
            self.model_path_var.set(path)

    def _apply_settings(self):
        """Apply GUI settings to config module."""
        cfg.CAPTURE_MODE = self.capture_mode_var.get()
        cfg.CAMERA_INDEX = int(self.camera_var.get().split()[0]) if self.camera_var.get() else 0
        cfg.CAPTURE_WIDTH = int(self.cap_width_var.get())
        cfg.CAPTURE_HEIGHT = int(self.cap_height_var.get())
        cfg.CAPTURE_FPS = int(float(self.cap_fps_var.get()))
        cfg.NDI_SOURCE_NAME = self.ndi_source_var.get()
        cfg.MONITOR_INDEX = int(float(self.monitor_var.get()))
        cfg.CAPTURE_SIZE = int(float(self.fov_var.get()))
        cfg.MODEL_PATH = self.model_path_var.get()
        cfg.HEAD_CLASS = int(float(self.head_class_var.get()))
        cfg.AIM_HEAD_Y_FRACTION = float(self.aim_y_var.get())
        cfg.MIN_CONFIDENCE = float(self.min_conf_var.get())
        cfg.DEBUG_PRINT = self.debug_var.get()
        cfg.SHOW_DEBUG_WINDOW = self.show_debug_var.get()
        cfg.IN_GAME_SENS = float(self.sens_var.get())
        cfg.BASE_SPEED = float(self.base_speed_var.get())
        cfg.DEADZONE_PX = float(self.deadzone_var.get())
        cfg.SMOOTHING_ALPHA = float(self.smooth_var.get())
        cfg.CLOSE_SPEED_MULT = float(self.close_speed_var.get())
        cfg.FAR_SPEED_MULT = float(self.far_speed_var.get())
        cfg.CLOSE_DISTANCE = float(self.close_dist_var.get())
        cfg.FAR_DISTANCE = float(self.far_dist_var.get())
        cfg.LOCK_LOST_FRAMES = int(float(self.lock_frames_var.get()))
        cfg.DETECTION_DELAY = float(self.det_delay_var.get())
        cfg.MOVE_TICK = float(self.move_tick_var.get())
        cfg.MAKCU_AUTO_RECONNECT = self.makcu_auto_var.get()
        cfg.MAKCU_DEBUG = self.makcu_debug_var.get()

    def _save_settings(self):
        """Save current settings to a file."""
        self._apply_settings()
        path = filedialog.asksaveasfilename(
            title="Save Settings",
            defaultextension=".txt",
            filetypes=[("Config files", "*.txt"), ("All files", "*.*")]
        )
        if path:
            try:
                with open(path, "w") as f:
                    for key, value in cfg.__dict__.items():
                        if not key.startswith("_") and key.isupper():
                            f.write(f"{key}={value}\n")
                self._log(f"Settings saved to {path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")

    def _load_settings(self):
        """Load settings from a file."""
        path = filedialog.askopenfilename(
            title="Load Settings",
            filetypes=[("Config files", "*.txt"), ("All files", "*.*")]
        )
        if path:
            try:
                with open(path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if "=" in line:
                            key, value = line.split("=", 1)
                            if hasattr(cfg, key):
                                # Try to preserve type
                                current = getattr(cfg, key)
                                if isinstance(current, bool):
                                    setattr(cfg, key, value.lower() in ("true", "1", "yes"))
                                elif isinstance(current, int):
                                    setattr(cfg, key, int(float(value)))
                                elif isinstance(current, float):
                                    setattr(cfg, key, float(value))
                                else:
                                    setattr(cfg, key, value.strip('"').strip("'"))
                self._refresh_ui_from_config()
                self._log(f"Settings loaded from {path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load: {e}")

    def _reset_settings(self):
        """Reset all settings to defaults."""
        if messagebox.askyesno("Reset", "Reset all settings to defaults?"):
            # Re-import config to get fresh defaults
            import importlib
            importlib.reload(cfg)
            self._refresh_ui_from_config()
            self._log("Settings reset to defaults")

    def _refresh_ui_from_config(self):
        """Update all UI elements from current config values."""
        self.capture_mode_var.set(cfg.CAPTURE_MODE)
        self.cap_width_var.set(str(cfg.CAPTURE_WIDTH))
        self.cap_height_var.set(str(cfg.CAPTURE_HEIGHT))
        self.cap_fps_var.set(str(cfg.CAPTURE_FPS))
        self.ndi_source_var.set(cfg.NDI_SOURCE_NAME)
        self.monitor_var.set(str(cfg.MONITOR_INDEX))
        self.fov_var.set(str(cfg.CAPTURE_SIZE))
        self.model_path_var.set(cfg.MODEL_PATH)
        self.head_class_var.set(str(cfg.HEAD_CLASS))
        self.aim_y_var.set(str(cfg.AIM_HEAD_Y_FRACTION))
        self.min_conf_var.set(str(cfg.MIN_CONFIDENCE))
        self.debug_var.set(cfg.DEBUG_PRINT)
        self.show_debug_var.set(cfg.SHOW_DEBUG_WINDOW)
        self.sens_var.set(str(cfg.IN_GAME_SENS))
        self.base_speed_var.set(str(cfg.BASE_SPEED))
        self.deadzone_var.set(str(cfg.DEADZONE_PX))
        self.smooth_var.set(str(cfg.SMOOTHING_ALPHA))
        self.close_speed_var.set(str(cfg.CLOSE_SPEED_MULT))
        self.far_speed_var.set(str(cfg.FAR_SPEED_MULT))
        self.close_dist_var.set(str(cfg.CLOSE_DISTANCE))
        self.far_dist_var.set(str(cfg.FAR_DISTANCE))
        self.lock_frames_var.set(str(cfg.LOCK_LOST_FRAMES))
        self.det_delay_var.set(str(cfg.DETECTION_DELAY))
        self.move_tick_var.set(str(cfg.MOVE_TICK))
        self.makcu_auto_var.set(cfg.MAKCU_AUTO_RECONNECT)
        self.makcu_debug_var.set(cfg.MAKCU_DEBUG)
        self._on_capture_mode_change()

    def _start_aimbot(self):
        """Start the aimbot."""
        if self.running:
            return

        self._apply_settings()

        try:
            from main import start_aimbot
            self.controller = start_aimbot()
            if self.controller:
                self.running = True
                self.start_btn.configure(state="disabled")
                self.stop_btn.configure(state="normal")
                self.status_label.configure(text="● Running", foreground="#00e676")
                self._log("Aimbot started!")
                self._poll_status()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start aimbot:\n{e}")
            self._log(f"ERROR: {e}")

    def _stop_aimbot(self):
        """Stop the aimbot."""
        if not self.running:
            return

        try:
            from main import stop_aimbot
            stop_aimbot(camera=None, controller=self.controller)
        except Exception as e:
            self._log(f"Stop error: {e}")

        self.running = False
        self.controller = None
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_label.configure(text="● Stopped", foreground="#ff1744")
        self.aim_label.configure(text="Aim: OFF", foreground="#888")
        self.fps_label.configure(text="FPS: --")
        self.det_label.configure(text="Detections: --")
        self._log("Aimbot stopped.")

    def _toggle_aim(self):
        """Toggle aim on/off."""
        if not self.running:
            return
        try:
            from main import toggle_aim
            state = toggle_aim()
            self.aim_active = state
            self.aim_label.configure(
                text=f"Aim: {'ON' if state else 'OFF'}",
                foreground="#00e676" if state else "#888"
            )
        except Exception as e:
            self._log(f"Toggle error: {e}")

    def _poll_status(self):
        """Poll aimbot status for GUI updates."""
        if not self.running:
            return

        try:
            from main import get_fps, is_aim_toggled, get_detection_info
            fps = get_fps()
            self.fps_label.configure(text=f"FPS: {fps:.1f}")

            aim_state = is_aim_toggled()
            self.aim_label.configure(
                text=f"Aim: {'ON' if aim_state else 'OFF'}",
                foreground="#00e676" if aim_state else "#888"
            )

            det_info = get_detection_info()
            if det_info:
                self.det_label.configure(
                    text=f"Detections: {det_info}",
                    foreground="#00e676" if "0 targ" not in det_info else "#888"
                )
        except Exception:
            pass

        self.root.after(200, self._poll_status)

    def _log(self, message):
        """Add a message to the log."""
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _on_close(self):
        """Handle window close."""
        if self.running:
            self._stop_aimbot()
        self.root.destroy()

    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()


if __name__ == "__main__":
    app = AimbotGUI()
    app.run()