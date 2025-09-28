from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from anomaly_detection.model import detect_triangle
from camera_feed.camera_receiver import run_listener
from industrial_predictor.predictor import train_model_and_get_predictions
import uvicorn
import asyncio
import base64
from multiprocessing import Process, Queue
import subprocess
import sys
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# A multiprocessing-safe queue to hold frames
frame_queue = Queue(maxsize=10)

@app.on_event("startup")
async def startup_event():
    """
    Start the camera transmitter and receiver in background processes.
    """
    # Start the transmitter using the current python executable
    transmitter_script_path = os.path.join("camera_feed", "camera_transmitter.py")
    subprocess.Popen([sys.executable, transmitter_script_path, "--port", "7000", "--dest", "127.0.0.1"])
    
    # Start the receiver in a separate process
    receiver_process = Process(target=run_listener, args=(7000, "127.0.0.1", frame_queue))
    receiver_process.daemon = True
    receiver_process.start()
    app.state.receiver_process = receiver_process

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "title": "Supply Chain Anomaly Detection"})

@app.get("/camera", response_class=HTMLResponse)
def camera_page(request: Request):
    return templates.TemplateResponse("camera.html", {"request": request, "title": "Camera Feed Anomaly Detection"})

@app.get("/industrial", response_class=HTMLResponse)
def industrial_page(request: Request):
    return templates.TemplateResponse("industrial.html", {"request": request, "title": "Industrial Anomaly Detection"})


@app.websocket("/ws/camera_feed")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    total_anomalies = 0

    while True:
        if not frame_queue.empty():
            frame_bytes = frame_queue.get_nowait()

            # Detect triangles with persistent IDs
            anomaly_detected, new_anomalies, bbox_coords, processed_frame = detect_triangle(frame_bytes)

            # Update total anomalies
            if new_anomalies > 0:
                total_anomalies += new_anomalies

            frame_base64 = base64.b64encode(processed_frame).decode("utf-8")

            await websocket.send_json({
                "frame": frame_base64,
                "anomaly": anomaly_detected,
                "count": new_anomalies,        # new anomalies in this frame
                "bboxes": bbox_coords,
                "total_anomalies": total_anomalies
            })

        await asyncio.sleep(1/60)  # ~60 FPS

@app.get("/industrial-predictions")
def industrial_predictions():
    """
    returns machine failure predictions
    """
    return train_model_and_get_predictions()

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


