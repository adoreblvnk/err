import random

def is_anomaly(frame_bytes: bytes) -> bool:
    """
    Dummy anomaly detection function.
    Replace this with a real model.
    """
    # For now, randomly flag 1 in 50 frames as an anomaly.
    return random.randint(0, 49) == 0
