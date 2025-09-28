#!/usr/bin/env python3
"""
cctv_udp_listener.py

Listener for the CCTV UDP simulator stream.
Receives UDP packets, reassembles frames, and displays them like a video.

Usage:
  python3 cctv_udp_listener.py --port 7000 --display
  python3 cctv_udp_listener.py --port 7000 --out-dir ./frames
"""

import argparse
import os
import socket
import struct
import time
import sys
from collections import defaultdict

import cv2
import numpy as np

MAGIC = b"CCTV"
HEADER_FMT = "!4sIHHd"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

class FrameBuffer:
    """Buffer to hold chunks until a frame is complete."""
    def __init__(self, total_chunks):
        self.total_chunks = total_chunks
        self.chunks = {}
        self.timestamp = None

    def add_chunk(self, idx, data, timestamp):
        self.chunks[idx] = data
        if self.timestamp is None:
            self.timestamp = timestamp

    def is_complete(self):
        return len(self.chunks) == self.total_chunks

    def assemble(self):
        return b"".join(self.chunks[i] for i in range(self.total_chunks))

def run_listener(port, bind_host, out_dir, display):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((bind_host, port))
    print(f"Listening on {bind_host}:{port}")

    frames = {}
    last_cleanup = time.time()
    while True:
        try:
            packet, addr = sock.recvfrom(65535)
            if len(packet) < HEADER_SIZE:
                continue
            magic, frame_seq, total_chunks, chunk_idx, ts = struct.unpack(
                HEADER_FMT, packet[:HEADER_SIZE]
            )
            if magic != MAGIC:
                continue

            payload = packet[HEADER_SIZE:]
            if frame_seq not in frames:
                frames[frame_seq] = FrameBuffer(total_chunks)
            fb = frames[frame_seq]
            fb.add_chunk(chunk_idx, payload, ts)

            if fb.is_complete():
                frame_bytes = fb.assemble()
                print(f"Received frame {frame_seq} ({len(frame_bytes)} bytes) from {addr}")

                # Save to disk if requested
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                    fname = os.path.join(out_dir, f"frame_{frame_seq:06d}.jpg")
                    with open(fname, "wb") as f:
                        f.write(frame_bytes)

                # Display with OpenCV if requested
                if display:
                    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        cv2.imshow("CCTV Stream", img)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            print("Exiting viewer.")
                            break

                # cleanup
                del frames[frame_seq]

            # periodic cleanup of stale frames
            if time.time() - last_cleanup > 5:
                cutoff = time.time() - 10
                to_delete = [
                    k for k, fb in frames.items()
                    if fb.timestamp and fb.timestamp < cutoff
                ]
                for k in to_delete:
                    del frames[k]
                last_cleanup = time.time()

        except KeyboardInterrupt:
            print("Interrupted by user - exiting.")
            break
        except Exception as e:
            print("Error:", e, file=sys.stderr)

    if display:
        cv2.destroyAllWindows()

def parse_args():
    p = argparse.ArgumentParser(description="CCTV UDP datastream listener.")
    p.add_argument("--port", type=int, required=True, help="UDP port to listen on.")
    p.add_argument("--bind-host", default="", help="Local bind host (default all).")
    p.add_argument("--out-dir", default=None, help="Directory to save frames as files.")
    p.add_argument("--display", action="store_true", help="Display frames as video (requires OpenCV).")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_listener(args.port, args.bind_host, args.out_dir, args.display)

