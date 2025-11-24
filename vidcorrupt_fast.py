#!/usr/bin/env python3
"""
Stream frames from ffmpeg -> process in-memory in batches -> pipe back to ffmpeg for encoding.
"""

import subprocess
import shutil
import os
import sys
import math
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from collections import deque
from vidcorrupt import init_worker, effect, SAMPLE_RATE

# ---------- helpers ----------
def run_cmd_get_output(cmd):
    return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()

def probe_video(path):
    """Return (width, height, fps, duration_seconds). Requires ffprobe."""
    # width
    w = run_cmd_get_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                            '-show_entries', 'stream=width', '-of', 'csv=p=0', path])
    h = run_cmd_get_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                            '-show_entries', 'stream=height', '-of', 'csv=p=0', path])
    # get r_frame_rate like "30000/1001" -> convert to float
    rfr = run_cmd_get_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                              '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', path])
    duration = run_cmd_get_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                   '-show_entries', 'format=duration', '-of', 'csv=p=0', path])
    
    w, h, rfr, duration = w.strip(','), h.strip(','), rfr.strip(','), duration.strip(',')

    def frac_to_float(frac):
        if '/' in frac:
            a,b = frac.split('/')
            return float(a)/float(b)
        return float(frac)
    return int(w), int(h), frac_to_float(rfr), float(duration)

# ---------- worker API ----------
def process_batch_worker(batch_tuple):
    """
    Worker input: (frames_np_list, first_frame_index, total_frame_count)
    frames_np_list: list of np.ndarray frames of shape (H, W, 3), dtype=uint8
    Returns: list of processed frames as uint8 arrays in same order
    """
    frames, first_idx, total_count = batch_tuple
    out_frames = []
    # lazy import of image libs/effect already in global state in worker
    for i, frame in enumerate(frames):
        # convert RGB uint8 -> image_data
        image_data = frame  # shape (H,W,3), dtype=uint8
        # Flatten to audio-like 1D float32
        audio_data = image_data.flatten().astype(np.float32, copy=False)
        minv = float(audio_data.min())
        maxv = float(audio_data.max())
        if maxv == minv:
            audio_data.fill(0.0)
        else:
            audio_data -= minv
            audio_data /= (maxv - minv)
            audio_data = 2.0 * audio_data - 1.0

        frame_no = first_idx + i
        # call your effect (assumes init_worker() prepared _GLOBAL_BOARD etc.)
        processed = effect(audio_data, SAMPLE_RATE, frame_no, total_count)

        # convert back to uint8 image
        processed_bytes = ((processed + 1.0) * 0.5 * 255.0).clip(0,255).astype(np.uint8)
        processed_image = processed_bytes.reshape(image_data.shape)
        out_frames.append(processed_image)
    return out_frames

# ---------- main streaming pipeline ----------
def pipeline_stream_in_memory(src_path, out_path,
                              batch_frames=12,
                              max_workers=None,
                              max_outstanding_batches=4):
    """
    Read raw frames from ffmpeg stdout in batches, submit to ProcessPoolExecutor,
    write processed raw frames into ffmpeg stdin for encoding.
    """
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        raise EnvironmentError("ffmpeg and ffprobe are required and must be in PATH")

    width, height, fps, duration = probe_video(src_path)
    total_frames = int(math.ceil(duration * fps))
    print(f"Video: {width}x{height} @ {fps:.3f} fps, duration {duration:.2f}s -> ~{total_frames} frames")

    frame_size = width * height * 3  # rgb24
    read_cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error',
        '-i', src_path,
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-vf', f'fps={fps}',
        '-'
    ]
    write_cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{width}x{height}',
        '-r', str(fps),
        '-i', '-',  # input from stdin
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        out_path
    ]

    # Start reader and writer ffmpeg processes
    reader = subprocess.Popen(read_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    writer = subprocess.Popen(write_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 1) // 2)

    executor = ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker)

    outstanding = deque()  # store tuples (future, first_frame_idx, batch_len)
    frame_idx = 0
    submitted_batches = 0

    try:
        while True:
            # read a batch of raw frames
            frames = []
            for _ in range(batch_frames):
                raw = reader.stdout.read(frame_size)
                if not raw or len(raw) < frame_size:
                    break
                # convert to numpy uint8 array shape (H,W,3)
                arr = np.frombuffer(raw, dtype=np.uint8)
                arr = arr.reshape((height, width, 3)).copy()  # copy to own memory
                frames.append(arr)
            if not frames:
                break  # EOF

            # submit batch to worker
            batch_input = (frames, frame_idx, total_frames)
            future = executor.submit(process_batch_worker, batch_input)
            outstanding.append((future, frame_idx, len(frames)))
            submitted_batches += 1
            frame_idx += len(frames)

            # If too many outstanding batches, wait for the oldest and write its results
            while len(outstanding) >= max_outstanding_batches:
                fut, start_idx, blen = outstanding.popleft()
                processed_frames = fut.result()  # will raise if worker failed
                # write processed frames to writer.stdin
                for pf in processed_frames:
                    writer.stdin.write(pf.tobytes())
                writer.stdin.flush()

        # all batches submitted; drain outstanding
        while outstanding:
            fut, start_idx, blen = outstanding.popleft()
            processed_frames = fut.result()
            for pf in processed_frames:
                writer.stdin.write(pf.tobytes())
            writer.stdin.flush()

    finally:
        # close writer stdin to let ffmpeg finalize file
        try:
            writer.stdin.close()
        except Exception:
            pass

        # wait for ffmpeg to finish
        writer.wait()
        reader.stdout.close()
        reader.wait()
        executor.shutdown(wait=True)

        # optional: print ffmpeg stderr for debugging
        # print(writer.stderr.read().decode())
        # print(reader.stderr.read().decode())

    print(f"Finished streaming. Output saved to: {out_path}")

# ---------- CLI ----------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stream-process video frames in-memory (no images on disk).")
    p.add_argument("input", help="input video")
    p.add_argument("output", help="output mp4")
    p.add_argument("--batch", type=int, default=8, help="frames per batch sent to workers (tune)")
    p.add_argument("--workers", type=int, default=None, help="number of worker processes")
    p.add_argument("--outstanding", type=int, default=3, help="max outstanding batches in memory")
    args = p.parse_args()
    pipeline_stream_in_memory(args.input, args.output,
                              batch_frames=args.batch,
                              max_workers=args.workers,
                              max_outstanding_batches=args.outstanding)
