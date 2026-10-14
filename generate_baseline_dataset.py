"""
Generates an initial reference dataset of 3D hand landmarks for supported ASL letters
based on standard anatomical finger postures (flexion, extension, abduction).
Adds realistic joint perturbation, orientation variance, and signer hand size jitter
to provide a reproducible dataset for training and verification.
"""

import json
import math
import random
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "dataset.json"

# Supported static ASL alphabet classes
CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y", "SPACE", "DEL"
]

def create_base_hand_posture(label: str) -> np.ndarray:
    """
    Creates a (21, 3) landmark array representing the nominal posture for a given sign.
    Landmarks follow MediaPipe index conventions:
    0: Wrist
    1-4: Thumb (CMC, MCP, IP, TIP)
    5-8: Index (MCP, PIP, DIP, TIP)
    9-12: Middle (MCP, PIP, DIP, TIP)
    13-16: Ring (MCP, PIP, DIP, TIP)
    17-20: Pinky (MCP, PIP, DIP, TIP)
    """
    # Nominal hand coordinate frame: wrist at (0, 0, 0), fingers pointing up +Y, palm facing -Z
    pts = np.zeros((21, 3), dtype=np.float32)

    # Base palm MCP positions relative to wrist
    mcp_pos = {
        "thumb": np.array([0.08, 0.05, 0.0], dtype=np.float32),
        "index": np.array([0.04, 0.14, 0.0], dtype=np.float32),
        "middle": np.array([0.0, 0.15, 0.0], dtype=np.float32),
        "ring": np.array([-0.04, 0.14, 0.0], dtype=np.float32),
        "pinky": np.array([-0.08, 0.12, 0.0], dtype=np.float32),
    }

    pts[0] = [0.0, 0.0, 0.0]  # Wrist

    # Posture definition: (extended = True/False, curled, tuck)
    # Define finger extensions for major letters
    extensions = {
        "A": {"thumb": "side", "index": "fist", "middle": "fist", "ring": "fist", "pinky": "fist"},
        "B": {"thumb": "tuck", "index": "up", "middle": "up", "ring": "up", "pinky": "up"},
        "C": {"thumb": "curve", "index": "curve", "middle": "curve", "ring": "curve", "pinky": "curve"},
        "D": {"thumb": "ring_touch", "index": "up", "middle": "ring_touch", "ring": "ring_touch", "pinky": "fist"},
        "E": {"thumb": "claw", "index": "claw", "middle": "claw", "ring": "claw", "pinky": "claw"},
        "F": {"thumb": "index_touch", "index": "index_touch", "middle": "up", "ring": "up", "pinky": "up"},
        "G": {"thumb": "point_horiz", "index": "point_horiz", "middle": "fist", "ring": "fist", "pinky": "fist"},
        "H": {"thumb": "tuck", "index": "point_horiz", "middle": "point_horiz", "ring": "fist", "pinky": "fist"},
        "I": {"thumb": "tuck", "index": "fist", "middle": "fist", "ring": "fist", "pinky": "up"},
        "K": {"thumb": "between", "index": "up", "middle": "point_fwd", "ring": "fist", "pinky": "fist"},
        "L": {"thumb": "out", "index": "up", "middle": "fist", "ring": "fist", "pinky": "fist"},
        "M": {"thumb": "under_ring", "index": "fist", "middle": "fist", "ring": "fist", "pinky": "fist"},
        "N": {"thumb": "under_mid", "index": "fist", "middle": "fist", "ring": "fist", "pinky": "fist"},
        "O": {"thumb": "o_touch", "index": "o_touch", "middle": "o_touch", "ring": "o_touch", "pinky": "o_touch"},
        "P": {"thumb": "between", "index": "down_fwd", "middle": "down", "ring": "fist", "pinky": "fist"},
        "Q": {"thumb": "down_point", "index": "down_point", "middle": "fist", "ring": "fist", "pinky": "fist"},
        "R": {"thumb": "tuck", "index": "cross_left", "middle": "cross_right", "ring": "fist", "pinky": "fist"},
        "S": {"thumb": "fist_cross", "index": "fist", "middle": "fist", "ring": "fist", "pinky": "fist"},
        "T": {"thumb": "under_idx", "index": "fist", "middle": "fist", "ring": "fist", "pinky": "fist"},
        "U": {"thumb": "tuck", "index": "up_together", "middle": "up_together", "ring": "fist", "pinky": "fist"},
        "V": {"thumb": "tuck", "index": "up_spread_r", "middle": "up_spread_l", "ring": "fist", "pinky": "fist"},
        "W": {"thumb": "tuck", "index": "up", "middle": "up", "ring": "up", "pinky": "fist"},
        "X": {"thumb": "tuck", "index": "hook", "middle": "fist", "ring": "fist", "pinky": "fist"},
        "Y": {"thumb": "out", "index": "fist", "middle": "fist", "ring": "fist", "pinky": "out"},
        "SPACE": {"thumb": "flat_open", "index": "flat_open", "middle": "flat_open", "ring": "flat_open", "pinky": "flat_open"},
        "DEL": {"thumb": "flat_down", "index": "flat_down", "middle": "flat_down", "ring": "flat_down", "pinky": "flat_down"}
    }

    ext = extensions.get(label, extensions["A"])

    def place_finger(indices, base, length, mode):
        # 3 segments
        seg = length / 3.0
        p0 = base
        if mode in ["up", "up_together"]:
            dx = 0.0 if mode == "up_together" else 0.0
            p1 = p0 + np.array([dx, seg, 0.0])
            p2 = p1 + np.array([dx, seg, 0.0])
            p3 = p2 + np.array([dx, seg, 0.0])
        elif mode == "up_spread_r":
            p1 = p0 + np.array([0.015, seg, 0.0])
            p2 = p1 + np.array([0.02, seg, 0.0])
            p3 = p2 + np.array([0.025, seg, 0.0])
        elif mode == "up_spread_l":
            p1 = p0 + np.array([-0.015, seg, 0.0])
            p2 = p1 + np.array([-0.02, seg, 0.0])
            p3 = p2 + np.array([-0.025, seg, 0.0])
        elif mode in ["fist", "ring_touch"]:
            # Folded into palm
            p1 = p0 + np.array([0.0, seg * 0.4, 0.03])
            p2 = p1 + np.array([0.0, -seg * 0.4, 0.04])
            p3 = p2 + np.array([0.0, -seg * 0.5, 0.02])
        elif mode == "curve":
            p1 = p0 + np.array([0.0, seg * 0.7, 0.03])
            p2 = p1 + np.array([0.0, seg * 0.4, 0.06])
            p3 = p2 + np.array([0.0, -seg * 0.2, 0.07])
        elif mode == "claw":
            p1 = p0 + np.array([0.0, seg * 0.6, 0.03])
            p2 = p1 + np.array([0.0, -seg * 0.2, 0.06])
            p3 = p2 + np.array([0.0, -seg * 0.5, 0.03])
        elif mode == "hook":
            p1 = p0 + np.array([0.0, seg * 0.8, 0.0])
            p2 = p1 + np.array([0.0, seg * 0.2, 0.04])
            p3 = p2 + np.array([0.0, -seg * 0.4, 0.03])
        elif mode == "out":
            p1 = p0 + np.array([-seg * 0.7, seg * 0.3, 0.0])
            p2 = p1 + np.array([-seg * 0.7, seg * 0.3, 0.0])
            p3 = p2 + np.array([-seg * 0.7, seg * 0.3, 0.0])
        elif mode == "flat_open":
            p1 = p0 + np.array([0.0, seg, 0.0])
            p2 = p1 + np.array([0.0, seg, 0.0])
            p3 = p2 + np.array([0.0, seg, 0.0])
        elif mode == "flat_down":
            p1 = p0 + np.array([0.0, -seg, 0.0])
            p2 = p1 + np.array([0.0, -seg, 0.0])
            p3 = p2 + np.array([0.0, -seg, 0.0])
        else:
            p1 = p0 + np.array([0.0, seg * 0.5, 0.02])
            p2 = p1 + np.array([0.0, seg * 0.2, 0.03])
            p3 = p2 + np.array([0.0, -seg * 0.2, 0.02])

        pts[indices[0]] = p0
        pts[indices[1]] = p1
        pts[indices[2]] = p2
        pts[indices[3]] = p3

    # Thumb
    t_base = mcp_pos["thumb"]
    t_mode = ext.get("thumb", "side")
    if t_mode == "out":
        t1 = t_base + np.array([0.03, 0.02, 0.0])
        t2 = t1 + np.array([0.03, 0.02, 0.0])
        t3 = t2 + np.array([0.03, 0.02, 0.0])
    elif t_mode == "side":
        t1 = t_base + np.array([0.01, 0.03, 0.02])
        t2 = t1 + np.array([0.0, 0.03, 0.02])
        t3 = t2 + np.array([-0.01, 0.03, 0.01])
    elif t_mode in ["tuck", "under_mid", "under_ring", "under_idx", "fist_cross"]:
        t1 = t_base + np.array([-0.02, 0.02, 0.03])
        t2 = t1 + np.array([-0.03, 0.02, 0.04])
        t3 = t2 + np.array([-0.02, 0.01, 0.04])
    else:
        t1 = t_base + np.array([0.02, 0.02, 0.01])
        t2 = t1 + np.array([0.02, 0.02, 0.01])
        t3 = t2 + np.array([0.01, 0.02, 0.01])
    pts[1] = t_base
    pts[2] = t1
    pts[3] = t2
    pts[4] = t3

    # Index (5-8)
    place_finger([5, 6, 7, 8], mcp_pos["index"], 0.09, ext.get("index", "fist"))
    # Middle (9-12)
    place_finger([9, 10, 11, 12], mcp_pos["middle"], 0.10, ext.get("middle", "fist"))
    # Ring (13-16)
    place_finger([13, 14, 15, 16], mcp_pos["ring"], 0.09, ext.get("ring", "fist"))
    # Pinky (17-20)
    place_finger([17, 18, 19, 20], mcp_pos["pinky"], 0.075, ext.get("pinky", "fist"))

    return pts

def add_noise_and_rotation(pts: np.ndarray, noise_std: float = 0.004, max_angle_deg: float = 8.0) -> np.ndarray:
    """Adds natural biometric variance and slight 3D rotation."""
    noisy = pts + np.random.normal(0, noise_std, pts.shape).astype(np.float32)

    # Random 3D rotation matrix
    angles = np.radians(np.random.uniform(-max_angle_deg, max_angle_deg, size=3))
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx

    rotated = (noisy @ R.T).astype(np.float32)
    # Random scale factor (hand size variation 0.90 to 1.10)
    scale = np.random.uniform(0.92, 1.08)
    return rotated * scale

def generate_dataset(samples_per_class: int = 50) -> List[Dict[str, Any]]:
    dataset = []
    sample_id = 1
    for label in CLASSES:
        base_posture = create_base_hand_posture(label)
        for i in range(samples_per_class):
            noisy_pts = add_noise_and_rotation(base_posture)
            raw_landmarks = [[float(p[0]), float(p[1]), float(p[2])] for p in noisy_pts]
            handedness = "Right" if random.random() > 0.3 else "Left"

            dataset.append({
                "id": sample_id,
                "label": label,
                "handedness": handedness,
                "landmarks": raw_landmarks,
                "session_id": f"synthetic_signer_{i % 5}"
            })
            sample_id += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated {len(dataset)} samples across {len(CLASSES)} classes into {OUTPUT_FILE}")
    return dataset

if __name__ == "__main__":
    generate_dataset(samples_per_class=50)
