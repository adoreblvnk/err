#!/usr/bin/env python3
"""
cctv_udp_listener.py

Listener for the CCTV UDP simulator stream.
Receives UDP packets, reassembles frames, and passes them to a queue.
"""

import argparse
import os
import socket
import struct
import time
import sys
from collections import defaultdict
import queue

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

def run_listener(port: int, bind_host: str, frame_queue: queue.Queue):
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
                try:
                    frame_queue.put_nowait(frame_bytes)
                except queue.Full:
                    # If the queue is full, drop the frame to avoid blocking
                    pass
                del frames[frame_seq]

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

def main():
    p = argparse.ArgumentParser(description="CCTV UDP datastream listener.")
    p.add_argument("--port", type=int, required=True, help="UDP port to listen on.")
    p.add_argument("--bind-host", default="", help="Local bind host (default all).")
    args = p.parse_args()
    
    # This part is for standalone execution, which we are not using in the app
    q = queue.Queue(maxsize=10)
    run_listener(args.port, args.bind_host, q)

if __name__ == "__main__":
    main()


