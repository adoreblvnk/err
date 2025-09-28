#!/usr/bin/env python3
"""
cctv_udp_simulator.py

Simulate a CCTV datastream by sending UDP packets from a port.
Usage examples:
  python3 cctv_udp_simulator.py --dest 127.0.0.1 --port 7000 --fps 10 --frames-dir /path/to/jpegs
  python3 cctv_udp_simulator.py --dest 192.168.1.100 --port 5000 --fps 15 --loop --mtu 1200

Behavior:
 - If --frames-dir is provided it will read images (jpg/png) sorted alphabetically and send each as a frame.
 - Otherwise it generates synthetic JPEG frames (requires Pillow).
 - Each frame is split into chunks not exceeding MTU (default 1400) and each UDP packet contains a small header:
     4 bytes magic b'CCTV'
     4 bytes frame_seq (unsigned int)
     2 bytes total_chunks (unsigned short)
     2 bytes chunk_idx (unsigned short) -- 0-based
     8 bytes timestamp (float packed as double)
   followed by payload bytes for that chunk.
 - You can control fps, mtu, loop, and sender bind address/port.
"""

import argparse
import os
import socket
import struct
import time
import glob
import sys
from camera_feed import generate_synthetic_jpeg_bytes 
try:
    from PIL import Image
    from io import BytesIO
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

MAGIC = b'CCTV'  # 4 bytes
HEADER_FMT = "!4sIHHd"  # magic, frame_seq, total_chunks, chunk_idx, timestamp (double)
HEADER_SIZE = struct.calcsize(HEADER_FMT)

def load_frames_from_dir(frames_dir):
    patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(os.path.join(frames_dir, p))))
    if not files:
        raise FileNotFoundError(f"No image files found in {frames_dir}")
    return files

def encode_image_to_jpeg_bytes(path):
    if PIL_AVAILABLE:
        im = Image.open(path).convert('RGB')
        buf = BytesIO()
        im.save(buf, format='JPEG', quality=80)
        return buf.getvalue()
    else:
        # Fallback: just read raw bytes (not ideal)
        with open(path, 'rb') as f:
            return f.read()

def chunk_and_send(sock, addr, frame_bytes, frame_seq, mtu):
    # payload space per packet
    max_payload = mtu - HEADER_SIZE
    if max_payload <= 0:
        raise ValueError("MTU too small for header")
    total = (len(frame_bytes) + max_payload - 1) // max_payload
    timestamp = time.time()
    for idx in range(total):
        start = idx * max_payload
        chunk = frame_bytes[start:start+max_payload]
        header = struct.pack(HEADER_FMT, MAGIC, frame_seq, total, idx, timestamp)
        packet = header + chunk
        sock.sendto(packet, addr)
    return total

def run_sender(dest, port, bind_host, bind_port, fps, mtu, frames_dir, loop, synth_size):
    addr = (dest, port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # optionally bind local port (so packets "come from" bind_port)
    if bind_host is not None or bind_port is not None:
        bhost = bind_host if bind_host is not None else ''
        bport = bind_port if bind_port is not None else 0
        sock.bind((bhost, bport))
    frame_paths = []
    use_frames = False
    if frames_dir:
        frame_paths = load_frames_from_dir(frames_dir)
        use_frames = True

    frame_seq = 0
    inter_frame = 1.0 / fps if fps > 0 else 0
    try:
        while True:
            t_start = time.time()
            if use_frames:
                path = frame_paths[frame_seq % len(frame_paths)]
                frame_bytes = encode_image_to_jpeg_bytes(path)
            else:
                frame_bytes = generate_synthetic_jpeg_bytes(synth_size[0], synth_size[1], frame_seq)

            total_chunks = chunk_and_send(sock, addr, frame_bytes, frame_seq, mtu)
            #print(f"Sent frame {frame_seq} as {total_chunks} chunks ({len(frame_bytes)} bytes) -> {dest}:{port}")
            frame_seq += 1
            if not loop and use_frames and frame_seq >= len(frame_paths):
                #print("Finished sending all frames (no-loop); exiting.")
                break
            # sleep to maintain fps
            t_elapsed = time.time() - t_start
            to_sleep = inter_frame - t_elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)
    except KeyboardInterrupt:
        print("Interrupted by user - exiting.")
    finally:
        sock.close()

def parse_args():
    p = argparse.ArgumentParser(description="CCTV UDP datastream simulator (sender).")
    p.add_argument("--dest", required=True, help="Destination IP to send UDP packets to.")
    p.add_argument("--port", type=int, required=True, help="Destination UDP port.")
    p.add_argument("--bind-host", default=None, help="Local bind host (optional).")
    p.add_argument("--bind-port", type=int, default=None, help="Local bind port (optional).")
    p.add_argument("--fps", type=float, default=10.0, help="Frames per second to send (default 10).")
    p.add_argument("--mtu", type=int, default=1400, help="Maximum UDP packet size (default 1400).")
    p.add_argument("--frames-dir", default=None, help="Directory of image frames (jpg/png). If omitted, generates synthetic frames.")
    p.add_argument("--loop", action="store_true", help="Loop frames directory when done (useful for tested camera stream).")
    p.add_argument("--synth-width", type=int, default=320, help="Width of synthetic frames (pixels).")
    p.add_argument("--synth-height", type=int, default=240, help="Height of synthetic frames (pixels).")
    p.add_argument("--start-seq", type=int, default=0, help="Start frame sequence number (default 0).")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        run_sender(args.dest, args.port, args.bind_host, args.bind_port, args.fps, args.mtu, args.frames_dir, args.loop, (args.synth_width, args.synth_height))
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(1)

