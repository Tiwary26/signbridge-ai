import math
from typing import List, Union, Dict, Any
import numpy as np

def normalize_landmarks(
    landmarks: Union[List[Dict[str, float]], List[List[float]], np.ndarray],
    handedness: str = "Right"
) -> List[float]:
    """
    Normalizes 21 3D hand landmarks for scale, translation, and handedness invariance.
    
    Args:
        landmarks: 21 landmarks either as list of dicts with 'x', 'y', 'z',
                   or list of [x, y, z] triples, or (21, 3) numpy array.
        handedness: 'Right' or 'Left'
        
    Returns:
        Flattened list of 63 normalized coordinates [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
    """
    # 1. Parse into numpy array (21, 3)
    coords = []
    if isinstance(landmarks, np.ndarray):
        coords = landmarks.copy()
        if coords.shape == (63,):
            coords = coords.reshape((21, 3))
    elif isinstance(landmarks, list):
        if len(landmarks) == 63 and isinstance(landmarks[0], (int, float)):
            coords = np.array(landmarks, dtype=np.float32).reshape((21, 3))
        elif len(landmarks) == 21:
            for item in landmarks:
                if isinstance(item, dict):
                    coords.append([item.get("x", 0.0), item.get("y", 0.0), item.get("z", 0.0)])
                elif isinstance(item, (list, tuple)):
                    coords.append([item[0], item[1], item[2] if len(item) > 2 else 0.0])
                elif hasattr(item, "x") and hasattr(item, "y"):
                    coords.append([item.x, item.y, getattr(item, "z", 0.0)])
                else:
                    raise ValueError(f"Unsupported landmark item format: {type(item)}")
            coords = np.array(coords, dtype=np.float32)
        else:
            raise ValueError(f"Expected 21 landmarks or 63 values, got length {len(landmarks)}")
    else:
        raise TypeError(f"Invalid landmarks type: {type(landmarks)}")

    if coords.shape != (21, 3):
        raise ValueError(f"Expected coords shape (21, 3), got {coords.shape}")

    # 2. Mirror Left hand to Right hand coordinate frame
    if handedness.lower() == "left":
        coords[:, 0] = -coords[:, 0]

    # 3. Translation invariance: shift wrist (index 0) to origin (0, 0, 0)
    wrist = coords[0].copy()
    translated = coords - wrist

    # 4. Scale invariance: determine palm scale
    # Euclidean distance from wrist (0) to middle finger MCP (landmark 9)
    # or max distance of all points to avoid div-by-zero
    mcp_dist = np.linalg.norm(translated[9])
    if mcp_dist < 1e-4:
        # Fallback to max distance
        dists = np.linalg.norm(translated, axis=1)
        mcp_dist = np.max(dists)
        if mcp_dist < 1e-4:
            mcp_dist = 1.0

    normalized = translated / mcp_dist

    # 5. Flatten to 63 floats
    return normalized.flatten().tolist()
