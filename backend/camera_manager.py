import cv2
import threading
import time
import os
import notification
import numpy as np

try:
    import mss
except ImportError:
    mss = None


# ============================================================
# Screen / Virtual Window configuration
# ============================================================
# The VMS/NVR window should show all 36 cameras in a 6x6 grid.
#
# Recommended:
#   - Put the VMS window on a 4K display: 3840x2160
#   - Use a 6x6 camera layout
#   - Keep the VMS window visible (not minimized)
#
# SCREEN_X/Y/W/H can be used to capture only the VMS area.
# If SCREEN_W/H are 0, the primary monitor is captured.
#
# Example for a full 4K monitor:
#   SCREEN_X=0
#   SCREEN_Y=0
#   SCREEN_W=3840
#   SCREEN_H=2160
#
SCREEN_X = int(os.getenv("SCREEN_X", "0"))
SCREEN_Y = int(os.getenv("SCREEN_Y", "0"))
SCREEN_W = int(os.getenv("SCREEN_W", "0"))
SCREEN_H = int(os.getenv("SCREEN_H", "0"))

GRID_COLS = int(os.getenv("CAMERA_GRID_COLS", "6"))
GRID_ROWS = int(os.getenv("CAMERA_GRID_ROWS", "6"))
TOTAL_GRID_CAMERAS = GRID_COLS * GRID_ROWS

# One camera is sent to AI at a time.
NORMAL_SCAN_INTERVAL = float(os.getenv("NORMAL_SCAN_INTERVAL", "1.0"))

# When a camera is suspicious, check it repeatedly before returning
# to the normal 36-camera scan.
SUSPECT_FRAMES = int(os.getenv("SUSPECT_FRAMES", "5"))
SUSPECT_INTERVAL = float(os.getenv("SUSPECT_INTERVAL", "0.20"))

# Optional upscaling for AI. 2.0 means 640x360 -> 1280x720, for example.
# IMPORTANT: upscaling does not create real camera detail. It only gives
# the model a larger image to work with.
AI_UPSCALE = float(os.getenv("AI_UPSCALE", "2.0"))

# Minimum image size sent to YOLO. The crop is enlarged only if smaller.
AI_MIN_WIDTH = int(os.getenv("AI_MIN_WIDTH", "640"))
AI_MIN_HEIGHT = int(os.getenv("AI_MIN_HEIGHT", "360"))

SNAPSHOT_DIR = "snapshots"
if not os.path.exists(SNAPSHOT_DIR):
    os.makedirs(SNAPSHOT_DIR)


class ScreenCapture:
    """
    Captures one VMS/NVR screen containing the 36-camera 6x6 grid.

    The screen is captured once per normal scan step, then only the
    required camera crop is sent to the detector. This avoids opening
    36 independent RTSP decoders in Python.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.sct = None
        self.monitor = None

        if mss is None:
            print(
                "[ScreenCapture] ERROR: 'mss' is not installed. "
                "Run: pip install mss"
            )
            return

        try:
            self.sct = mss.mss()
            self._configure_monitor()
            print(
                f"[ScreenCapture] Ready: "
                f"{self.monitor['width']}x{self.monitor['height']} "
                f"at ({self.monitor['left']},{self.monitor['top']})"
            )
        except Exception as e:
            print(f"[ScreenCapture] Initialization failed: {e}")
            self.sct = None

    def _configure_monitor(self):
        if self.sct is None:
            return

        if SCREEN_W > 0 and SCREEN_H > 0:
            self.monitor = {
                "left": SCREEN_X,
                "top": SCREEN_Y,
                "width": SCREEN_W,
                "height": SCREEN_H,
            }
        else:
            # Primary monitor.
            self.monitor = self.sct.monitors[1]

    def capture(self):
        """Capture the configured VMS display region."""
        if self.sct is None or self.monitor is None:
            return None

        try:
            with self.lock:
                shot = self.sct.grab(self.monitor)

            # MSS returns BGRA; OpenCV expects BGR.
            frame = np.asarray(shot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
        except Exception as e:
            print(f"[ScreenCapture] Capture failed: {e}")
            return None

    def get_camera_crop(self, full_frame, camera_number):
        """
        Split the screen into a 6x6 grid and return one camera image.

        camera_number is 1..36:
            1  2  3  4  5  6
            7  8  9 10 11 12
            ...
            31 32 33 34 35 36
        """
        if full_frame is None:
            return None

        if camera_number < 1 or camera_number > TOTAL_GRID_CAMERAS:
            return None

        height, width = full_frame.shape[:2]

        col = (camera_number - 1) % GRID_COLS
        row = (camera_number - 1) // GRID_COLS

        x1 = int(col * width / GRID_COLS)
        x2 = int((col + 1) * width / GRID_COLS)
        y1 = int(row * height / GRID_ROWS)
        y2 = int((row + 1) * height / GRID_ROWS)

        crop = full_frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        return crop

    def prepare_for_ai(self, crop):
        """
        Upscale a camera crop when useful.

        We do not blindly resize every image. We only enlarge it when it
        is below the configured AI minimum or AI_UPSCALE requests it.
        """
        if crop is None or crop.size == 0:
            return None

        h, w = crop.shape[:2]

        target_w = max(w, AI_MIN_WIDTH)
        target_h = max(h, AI_MIN_HEIGHT)

        if AI_UPSCALE > 1.0:
            target_w = max(target_w, int(w * AI_UPSCALE))
            target_h = max(target_h, int(h * AI_UPSCALE))

        if target_w == w and target_h == h:
            return crop

        return cv2.resize(
            crop,
            (target_w, target_h),
            interpolation=cv2.INTER_CUBIC,
        )

    def release(self):
        if self.sct is not None:
            try:
                self.sct.close()
            except Exception:
                pass


class Camera:
    """
    Logical camera used by the existing application/database.

    The source is retained for database compatibility, but frames now
    come from the 6x6 screen grid instead of opening an RTSP connection.
    """

    def __init__(self, id, source, name="Camera", is_active=True):
        self.id = id
        self.source = source
        self.name = name
        self.is_active = is_active

        self.lock = threading.Lock()
        self.running = True

        self.last_frame = None
        self.last_annotated_frame = None

        self.detection_persistence = {}
        self.event_cooldowns = {}

        self.frame_count = 0
        self.status = "connecting"
        self.consecutive_failures = 0

        # No VideoCapture here. The whole screen is captured by
        # CameraManager.screen_capture.
        self.cap = None

    def read(self):
        """Compatibility method. Actual screen capture is handled by manager."""
        if not self.is_active or not self.running:
            return False, None

        if self.last_frame is None:
            return False, None

        return True, self.last_frame.copy()

    def release(self):
        self.running = False
        self.status = "disconnected"


class CameraManager:
    def __init__(self, detector):
        self.cameras = {}
        self.detector = detector
        self.running = True
        self.alert_callback = None

        self.screen_capture = ScreenCapture()

        # Camera IDs in the database may not be exactly 1..36.
        # We map the first 36 active logical cameras to screen positions.
        self.screen_camera_map = {}

        # Load persisted cameras from database.
        self.load_cameras_from_db()

        # Start background detection thread.
        self.thread = threading.Thread(
            target=self._detection_loop,
            daemon=True
        )
        self.thread.start()

    def load_cameras_from_db(self):
        """Load cameras from database as logical 6x6 screen cameras."""
        try:
            import database

            db_cams = database.get_all_db_cameras()

            for db_cam in db_cams:
                cam_id = db_cam["id"]
                name = db_cam["name"]
                source = db_cam["source"]
                is_active = db_cam["is_active"] == 1

                if cam_id in self.cameras:
                    self.cameras[cam_id].name = name
                    self.cameras[cam_id].source = source
                    self.cameras[cam_id].is_active = is_active
                else:
                    cam = Camera(
                        cam_id,
                        source,
                        name,
                        is_active
                    )
                    self.cameras[cam_id] = cam

            self._rebuild_screen_camera_map()

            print(f"Loaded {len(db_cams)} cameras from database.")
            print(
                f"Screen mode: {GRID_COLS}x{GRID_ROWS} = "
                f"{TOTAL_GRID_CAMERAS} camera positions"
            )

        except Exception as e:
            print(f"Error loading cameras from database: {e}")

    def _rebuild_screen_camera_map(self):
        """
        Map active database cameras to positions 1..36.

        The database ID is NOT assumed to be the screen position.
        Active cameras are ordered by database ID.
        """
        self.screen_camera_map = {}

        active_cams = [
            cam for cam in self.cameras.values()
            if cam.is_active and cam.running
        ]

        active_cams.sort(key=lambda c: c.id)

        for position, cam in enumerate(active_cams[:TOTAL_GRID_CAMERAS], start=1):
            self.screen_camera_map[position] = cam.id

        if len(active_cams) > TOTAL_GRID_CAMERAS:
            print(
                f"[ScreenCapture] WARNING: {len(active_cams)} active cameras "
                f"but the screen grid has only {TOTAL_GRID_CAMERAS} positions."
            )

    def add_camera(self, source, name="Camera"):
        """Add a logical camera to the database."""
        import database

        cam_id = database.add_db_camera(name, source, 1)
        cam = Camera(cam_id, source, name, True)
        self.cameras[cam_id] = cam

        self._rebuild_screen_camera_map()
        return cam_id

    def remove_camera(self, camera_id):
        """Remove a camera."""
        import database

        if camera_id in self.cameras:
            self.cameras[camera_id].release()
            del self.cameras[camera_id]
            database.delete_db_camera(camera_id)

            self._rebuild_screen_camera_map()
            return True

        return False

    def toggle_camera(self, camera_id, is_active):
        """Enable or disable a logical camera."""
        import database

        if camera_id in self.cameras:
            cam = self.cameras[camera_id]
            cam.is_active = is_active

            database.set_db_camera_status(camera_id, is_active)

            if not is_active:
                cam.release()
            else:
                cam.running = True
                cam.status = "connecting"
                cam.consecutive_failures = 0

            self._rebuild_screen_camera_map()
            return True

        return False

    def get_frame(self, camera_id):
        """Get the latest annotated/captured frame for a camera."""
        if camera_id not in self.cameras:
            return None

        cam = self.cameras[camera_id]

        with cam.lock:
            if cam.last_annotated_frame is not None:
                return cam.last_annotated_frame.copy()

            if cam.last_frame is not None:
                return cam.last_frame.copy()

        return None

    def _capture_and_crop(self, screen_position):
        """Capture the VMS screen and extract one camera position."""
        full_screen = self.screen_capture.capture()

        if full_screen is None:
            return None

        crop = self.screen_capture.get_camera_crop(
            full_screen,
            screen_position
        )

        return self.screen_capture.prepare_for_ai(crop)

    def _run_detection(self, cam, screen_position):
        """
        Capture one grid position and run YOLO on it.

        Returns True when the camera was successfully processed.
        """
        frame = self._capture_and_crop(screen_position)

        if frame is None:
            cam.status = "disconnected"
            cam.consecutive_failures += 1
            return False

        cam.status = "connected"
        cam.consecutive_failures = 0
        cam.frame_count += 1

        annotated_frame, detections = self.detector.process_frame(frame)

        with cam.lock:
            cam.last_frame = frame
            cam.last_annotated_frame = annotated_frame

        if detections:
            self._process_detections(
                cam,
                detections,
                annotated_frame
            )
        else:
            cam.detection_persistence = {}

        return True

    def _normal_scan(self):
        """
        Normal mode:
        C1 -> C2 -> ... -> C36

        Only ONE camera is sent to YOLO in each step.
        """
        self._rebuild_screen_camera_map()

        if not self.screen_camera_map:
            time.sleep(1.0)
            return

        for screen_position in range(1, TOTAL_GRID_CAMERAS + 1):
            if not self.running:
                return

            camera_id = self.screen_camera_map.get(screen_position)

            if camera_id is None:
                continue

            cam = self.cameras.get(camera_id)

            if not cam or not cam.running or not cam.is_active:
                continue

            processed = self._run_detection(
                cam,
                screen_position
            )

            # One-second pacing is intentional:
            # 36 cameras = approximately one full scan every 36 seconds.
            if processed:
                time.sleep(NORMAL_SCAN_INTERVAL)

    def _focus_suspect_camera(self, cam, screen_position):
        """
        Suspicious camera mode:
        analyze the same camera repeatedly before returning to normal scan.
        """
        print(
            f"[Camera {cam.id}] Suspicious result. "
            f"Focusing camera for {SUSPECT_FRAMES} frames..."
        )

        # Reset persistence before focused verification.
        cam.detection_persistence = {}

        for _ in range(SUSPECT_FRAMES):
            if not self.running or not cam.running or not cam.is_active:
                return

            self._run_detection(cam, screen_position)
            time.sleep(SUSPECT_INTERVAL)

    def _detection_loop(self):
        """
        Main screen-based detection loop.

        Architecture:
          VMS 6x6 screen
             ↓
          capture screen
             ↓
          crop one camera
             ↓
          YOLO
             ↓
          suspicious?
             ↓ yes
          focus same camera
             ↓
          return to normal scan
        """
        if self.screen_capture.sct is None:
            print(
                "[CameraManager] Screen capture is not available. "
                "Install mss and restart the backend."
            )

            while self.running:
                time.sleep(5)

            return

        print(
            "[CameraManager] Screen-based detection started. "
            f"Grid={GRID_COLS}x{GRID_ROWS}, "
            f"normal_interval={NORMAL_SCAN_INTERVAL}s"
        )

        while self.running:
            self._rebuild_screen_camera_map()

            if not self.screen_camera_map:
                time.sleep(2)
                continue

            for screen_position in range(1, TOTAL_GRID_CAMERAS + 1):
                if not self.running:
                    break

                camera_id = self.screen_camera_map.get(screen_position)

                if camera_id is None:
                    continue

                cam = self.cameras.get(camera_id)

                if not cam or not cam.running or not cam.is_active:
                    continue

                processed = self._run_detection(
                    cam,
                    screen_position
                )

                if not processed:
                    time.sleep(0.1)
                    continue

                # Determine whether this camera became suspicious.
                suspicious = self._is_suspicious(cam)

                if suspicious:
                    self._focus_suspect_camera(
                        cam,
                        screen_position
                    )

                time.sleep(NORMAL_SCAN_INTERVAL)

    def _is_suspicious(self, cam):
        """
        Returns True when the current camera has a detection that
        passed the configured confidence threshold but has not yet
        necessarily reached the final alert threshold.
        """
        if not cam.detection_persistence:
            return False

        try:
            import database

            conf_threshold = float(
                database.get_setting(
                    "ai_confidence_threshold",
                    "0.25"
                )
            )
            persistence_limit = int(
                database.get_setting(
                    "ai_persistence_frames",
                    "2"
                )
            )
        except Exception:
            conf_threshold = 0.25
            persistence_limit = 2

        # If persistence is already at the alert level, _process_detections
        # has handled the alert. We still return True for focused verification.
        for count in cam.detection_persistence.values():
            if count >= 1 and conf_threshold < 0.60:
                return True

        return False

    def _process_detections(self, cam, detections, annotated_frame):
        current_classes = set()

        # Get dynamic settings from database.
        import database

        try:
            conf_threshold = float(
                database.get_setting(
                    "ai_confidence_threshold",
                    "0.25"
                )
            )
            persistence_limit = int(
                database.get_setting(
                    "ai_persistence_frames",
                    "2"
                )
            )
            cooldown_seconds = float(
                database.get_setting(
                    "ai_cooldown_seconds",
                    "5"
                )
            )
        except Exception:
            conf_threshold = 0.25
            persistence_limit = 2
            cooldown_seconds = 5

        for det in detections:
            cls_name = det["class"]
            conf = det["confidence"]

            if conf < conf_threshold:
                continue

            current_classes.add(cls_name)

            # Increment persistence.
            cam.detection_persistence[cls_name] = (
                cam.detection_persistence.get(cls_name, 0) + 1
            )

            # Trigger alert.
            if (
                cam.detection_persistence[cls_name] >= persistence_limit
                or conf > 0.6
            ):
                current_time = time.time()
                last_time = cam.event_cooldowns.get(cls_name, 0)

                if current_time - last_time > cooldown_seconds:
                    self._save_event(
                        cam,
                        cls_name,
                        conf,
                        annotated_frame
                    )

                    cam.event_cooldowns[cls_name] = current_time

        # Cleanup persistence.
        for existing_cls in list(cam.detection_persistence.keys()):
            if existing_cls not in current_classes:
                cam.detection_persistence[existing_cls] = 0

    def _save_event(self, cam, cls_name, conf, frame):
        # Save snapshot.
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = (
            f"event_{timestamp_str}_{cam.id}_{cls_name}.jpg"
        )
        filepath = os.path.join(
            SNAPSHOT_DIR,
            filename
        )

        cv2.imwrite(filepath, frame)

        # Database.
        snapshot_url = f"/snapshots/{filename}"
        import database

        event_id, timestamp = database.add_event(
            cls_name,
            conf,
            snapshot_url
        )

        print(
            f"[Cam {cam.id}] Detected {cls_name} "
            f"({conf:.2f})"
        )

        # Trigger WebSocket callback if registered.
        if self.alert_callback:
            try:
                self.alert_callback({
                    "id": event_id,
                    "timestamp": timestamp,
                    "type": cls_name,
                    "confidence": conf,
                    "snapshot": snapshot_url,
                    "camera_name": cam.name
                })
            except Exception as e:
                print(
                    f"Error in alert callback: {e}"
                )

        # Notifications.
        threading.Thread(
            target=notification.trigger_notifications,
            args=(cls_name, conf, filepath),
            daemon=True
        ).start()

    def release(self):
        """Stop screen capture and all logical cameras."""
        self.running = False

        for cam in self.cameras.values():
            cam.release()

        self.screen_capture.release()
