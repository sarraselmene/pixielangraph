import math
import numpy as np
from collections import deque
from dataclasses import dataclass

@dataclass
class BehaviorState:
    posture: str           # "sitting", "standing", "slouching"
    action: str            # "none", "hand_raised", "fidgeting", "bouncing"
    posture_conf: float
    action_conf: float


class UpperBodyBehaviorClassifier:
    """
    Temporal behavior classifier using ONLY upper-body keypoints.
    Designed to work without legs/knees, making it robust for classroom/desk scenarios.
    
    Key Features:
    - Temporal Smoothing: Uses a sliding window (e.g., 30 frames) to compute variances and velocities.
    - Relative Distances: Uses the global desk Y-coordinate and shoulder width (sw) to normalize.
    - Slouching Detection: Uses Head-Desk ratio (NDR) and Nose-to-Shoulder geometry.
    """
    
    def __init__(self, fps=30.0, window_size_frames=60):
        self.fps = fps
        self.window_size = window_size_frames
        
        # Temporal buffers for the key upper-body signals
        # Fidgeting uses window_size_frames (~2 seconds)
        # Slouching uses a much longer buffer (~5 seconds) for consistency
        extended_window = int(fps * 5) # 150 frames
        
        self.history = {
            "head_y": deque(maxlen=window_size_frames),
            "shoulder_y": deque(maxlen=window_size_frames),
            "shoulder_x": deque(maxlen=window_size_frames),
            "wrist_y": deque(maxlen=window_size_frames),
            "wrist_x": deque(maxlen=window_size_frames),
            "ndr": deque(maxlen=extended_window), 
            "nose_sh_diff": deque(maxlen=extended_window)
        }
        
        # Baselines
        self.baseline_shoulder_y = None
        
        # Current State
        self.current_posture = "sitting"
        self.posture_counter = 0

    def update(self, keypoints, global_desk_y, shoulder_width):
        """
        keypoints: dictionary of normalized { 'nose': (x, y), 'left_shoulder': (x,y), ... }
        global_desk_y: The Y-coordinate of the desk line.
        shoulder_width: The pixel distance between left and right shoulders (used as a ruler).
        """
        # 1. Extract & Sanitize Upper Body Features
        nose = keypoints.get("nose")
        l_shoulder = keypoints.get("left_shoulder")
        r_shoulder = keypoints.get("right_shoulder")
        l_wrist = keypoints.get("left_wrist")
        r_wrist = keypoints.get("right_wrist")
        
        # Mid-shoulder
        if l_shoulder and r_shoulder:
            sh_x = (l_shoulder[0] + r_shoulder[0]) / 2.0
            sh_y = (l_shoulder[1] + r_shoulder[1]) / 2.0
        else:
            sh_x, sh_y = None, None

        # Best wrist (highest on screen -> smallest Y)
        wrist_y_list = [w[1] for w in (l_wrist, r_wrist) if w]
        best_wrist_y = min(wrist_y_list) if wrist_y_list else None
        
        wrist_x_list = [w[0] for w in (l_wrist, r_wrist) if w]
        best_wrist_x = wrist_x_list[0] if wrist_x_list else None 

        # Update Baseline
        if sh_y is not None:
            if self.baseline_shoulder_y is None or self.current_posture == "sitting":
                prev = self.baseline_shoulder_y if self.baseline_shoulder_y else sh_y
                self.baseline_shoulder_y = 0.95 * prev + 0.05 * sh_y

        sw = max(shoulder_width, 1.0) 

        # 2. Push to Temporal Buffers (Normalized by shoulder width)
        if nose: self.history["head_y"].append(nose[1] / sw)
        if sh_y: self.history["shoulder_y"].append(sh_y / sw)
        if sh_x: self.history["shoulder_x"].append(sh_x / sw)
        if best_wrist_y: self.history["wrist_y"].append(best_wrist_y / sw)
        if best_wrist_x: self.history["wrist_x"].append(best_wrist_x / sw)
        
        if nose and global_desk_y:
            ndr = (global_desk_y - nose[1]) / sw
            self.history["ndr"].append(ndr)
        
        if nose and sh_y:
            diff = (sh_y - nose[1]) / sw
            self.history["nose_sh_diff"].append(diff)

        # 3. Predict Posture (Sitting vs Standing vs Slouching)
        posture, p_conf = self._predict_posture_and_slouch(nose, sh_y, sw)
        
        # Hysteresis for major state changes (Wait for ~0.5s of consistency)
        if posture != self.current_posture:
            self.posture_counter += 1
            if self.posture_counter >= 15: 
                self.current_posture = posture
                self.posture_counter = 0
        else:
            self.posture_counter = 0

        # 4. Predict Short-Term Action
        action, a_conf = self._predict_action(sh_y, best_wrist_y, sw)

        return BehaviorState(
            posture=self.current_posture,
            action=action,
            posture_conf=p_conf,
            action_conf=a_conf
        )

    def _predict_posture_and_slouch(self, nose, sh_y, sw):
        ndr_history = list(self.history["ndr"])
        nsdiff_history = list(self.history["nose_sh_diff"])
        
        # --- A. Standing Detection (Prioritized) ---
        if len(ndr_history) >= 15:
            # Use short-medium average for standing (more responsive than slouch)
            current_ndr = np.mean(ndr_history[-15:])
            if current_ndr > 1.35:
                return "standing", min(1.0, (current_ndr - 1.1) / 0.4)

        # --- B. Slouching Detection (Robust Sliding Window) ---
        slouch_score = 0.0
        
        # 1. Nose-Desk Ratio contribution (Requires 2+ seconds of low head)
        if len(ndr_history) >= 60:
            long_ndr = np.mean(ndr_history) # Full 2-5s window
            if long_ndr < 0.8:
                slouch_score += 0.4 * min(1.0, (0.8 - long_ndr) / 0.3)

        # 2. Geometric Head-Between-Shoulders contribution (Sliding window)
        if len(nsdiff_history) >= 60:
            long_nsdiff = np.mean(nsdiff_history)
            if long_nsdiff < 0.25: 
                slouch_score += 0.6 * min(1.0, (0.25 - long_nsdiff) / 0.2)
        elif nose and sh_y:
            # Fallback for very short tracks
            nose_sh_diff = (sh_y - nose[1]) / sw
            if nose_sh_diff < 0.22:
                slouch_score += 0.3

        if slouch_score > 0.50:
            return "slouching", min(1.0, slouch_score)

        return "sitting", 0.8

    def _predict_action(self, current_sh_y, current_wrist_y, sw):
        # 1. Hand Raise
        if current_sh_y and current_wrist_y:
            if (current_sh_y - current_wrist_y) / sw > 0.22:
                return "hand_raised", 0.9

        # 2. Extract variances for dynamic movements
        sh_y_var = self._safe_var(self.history["shoulder_y"])
        sh_x_var = self._safe_var(self.history["shoulder_x"])
        wr_y_var = self._safe_var(self.history["wrist_y"])
        wr_x_var = self._safe_var(self.history["wrist_x"])
        
        # Bouncing (High vertical variance on shoulders - smoothed over window)
        if sh_y_var > 0.0022:
            return "bouncing", min(1.0, sh_y_var * 150)

        # Fidgeting (Requires consistent movement over the 2s window)
        if sh_x_var > 0.0015 or wr_x_var > 0.0045 or wr_y_var > 0.0045:
            return "fidgeting", 0.75

        return "none", 0.0

    def _safe_var(self, deque_buf):
        if len(deque_buf) < self.window_size:
            return 0.0
        arr = np.array(deque_buf)
        # Detrend to remove slow drift (only capture rapid kinetic movements)
        x = np.arange(len(arr))
        detrended = arr - np.polyval(np.polyfit(x, arr, 1), x)
        return np.var(detrended)

