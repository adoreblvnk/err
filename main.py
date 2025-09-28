from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import asyncio
import base64
from multiprocessing import Process, Queue
import subprocess

from anomaly_detection.model import is_anomaly
from camera_feed.camera_receiver import run_listener

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
    # Start the transmitter
    subprocess.Popen(["python3", "camera_feed/camera_transmitter.py", "--port", "7000", "--dest", "127.0.0.1"])
    
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

@app.websocket("/ws/camera_feed")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            if not frame_queue.empty():
                frame_bytes = frame_queue.get_nowait()
                
                # --- Anomaly Detection ---
                anomaly_detected = is_anomaly(frame_bytes)
                # -------------------------

                frame_base64 = base64.b64encode(frame_bytes).decode("utf-8")
                
                await websocket.send_json({
                    "frame": frame_base64,
                    "anomaly": anomaly_detected
                })
            await asyncio.sleep(1/60) # ~60fps
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error in websocket: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


