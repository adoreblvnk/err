import cv2
import numpy as np
import time

# Global storage for tracked triangles
tracked_triangles = {}
next_triangle_id = 0

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

def detect_triangle(frame_bytes, distance_threshold=30, max_seconds_lost=1.0, edge_margin=5):
    """
    Detect triangles, track anomalies, draw bounding boxes + labels.
    Ignores triangles that touch the edges of the frame.
    """
    global tracked_triangles, next_triangle_id
    current_time = time.time()

    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    frame_height, frame_width = frame.shape[:2]

    # Mask for colored areas
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 50, 50), (180, 255, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = False
    new_anomalies = 0
    bboxes = []

    # Remove old triangles
    to_remove = [tid for tid, info in tracked_triangles.items()
                 if current_time - info["last_seen"] > max_seconds_lost]
    for tid in to_remove:
        tracked_triangles.pop(tid)

    frame_boxes = []
    for cnt in contours:
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 3:
            x, y, w, h = cv2.boundingRect(approx)

            # Skip triangles touching the edges
            if x <= edge_margin or y <= edge_margin or (x+w) >= (frame_width - edge_margin) or (y+h) >= (frame_height - edge_margin):
                continue

            cx = int(x + w/2)
            cy = int(y + h/2)
            frame_boxes.append({"center": (cx, cy), "bbox": (x, y, x+w, y+h)})

    # Merge overlapping boxes within frame (IoU > 0.5)
    merged_boxes = []
    for box in frame_boxes:
        keep = True
        for mbox in merged_boxes:
            if iou(box["bbox"], mbox["bbox"]) > 0.5:
                keep = False
                break
        if keep:
            merged_boxes.append(box)

    for box in merged_boxes:
        cx, cy = box["center"]
        x1, y1, x2, y2 = box["bbox"]
        matched_id = None
        for tid, info in tracked_triangles.items():
            px, py = info["center"]
            if np.linalg.norm(np.array([cx, cy]) - np.array([px, py])) <= distance_threshold:
                matched_id = tid
                break

        if matched_id is None:
            tid = next_triangle_id
            next_triangle_id += 1
            tracked_triangles[tid] = {"center": (cx, cy), "bbox": (x1, y1, x2, y2), "last_seen": current_time}
            new_anomalies += 1
        else:
            tracked_triangles[matched_id]["center"] = (cx, cy)
            tracked_triangles[matched_id]["bbox"] = (x1, y1, x2, y2)
            tracked_triangles[matched_id]["last_seen"] = current_time

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # Draw label above box
        cv2.putText(frame, "Triangle", (x1, max(0, y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        bboxes.append((x1, y1, x2, y2))
        detected = True

    _, buffer = cv2.imencode(".jpg", frame)
    processed_frame = buffer.tobytes()

    return detected, new_anomalies, bboxes, processed_frame
