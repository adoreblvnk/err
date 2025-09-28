import cv2
import numpy as np

def detect_triangle(frame_bytes):
    """
    Detects triangles in a video frame and highlights them.
    Returns:
      detected (bool) → True if at least one triangle found
      bboxes (list) → bounding boxes [(x1, y1, x2, y2), ...]
      processed_frame (bytes) → JPEG encoded frame with drawings
    """
    # Decode bytes → OpenCV image
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask everything except "colored" regions (ignore gray bg)
    mask = cv2.inRange(hsv, (0, 50, 50), (180, 255, 255))

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = False
    bboxes = []

    for cnt in contours:
        # Approximate polygon
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 3:  # triangle
            detected = True
            x, y, w, h = cv2.boundingRect(approx)
            bboxes.append((x, y, x+w, y+h))

            # Draw thick cyan contour
            cv2.drawContours(frame, [approx], 0, (0, 255, 255), 4)
            # Draw bounding box in red
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            # Add label
            cv2.putText(frame, "Triangle", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Encode back to bytes
    _, buffer = cv2.imencode(".jpg", frame)
    processed_frame = buffer.tobytes()

    return detected, bboxes, processed_frame
