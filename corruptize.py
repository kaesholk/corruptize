import subprocess
import os
import shutil
import json
from pedalboard import Pedalboard, Phaser
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import gc

def effect(audio_data, fs, frame_no, frame_count):
    coef = float(frame_no) / float(frame_count)
    board = Pedalboard([Phaser(
        rate_hz= coef * 1,
        mix=1
    )])
    return board(audio_data, fs)

def extract_frames(file_path, output_dir, fps):
    os.makedirs(output_dir, exist_ok=True)
    ffmpeg_command = [
        'ffmpeg.exe',
        '-i', file_path,
        '-vf', f"fps={fps}",
        os.path.join(output_dir, 'frame_%04d.tiff')
    ]
    subprocess.run(ffmpeg_command, check=True)
    print(f"Extracted {len(os.listdir(output_dir))} frames at {fps} fps\n")

def reconstruct_video(image_directory, output_path, fps):
    if not os.path.exists(image_directory):
        raise ValueError("Image directory does not exist.")

    command = [
        'ffmpeg.exe',
        '-framerate', str(fps),
        '-i', os.path.join(image_directory, 'frame_%04d.png'), 
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_path
    ]

    try:
        # Run the ffmpeg command
        subprocess.run(command, check=True)
        print(f"Video saved as {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")

def process_frame(image_path, output_dir, frame_count):
    base_name, _ = os.path.splitext(os.path.basename(image_path))
    output_path = os.path.join(output_dir, f"{base_name}.png")
    if (os.path.isfile(output_path)): return

    image = Image.open(image_path)
    image_data = np.array(image)

    # Flatten image data to 1D ~ audio representation
    audio_data = image_data.flatten().astype(np.float32)

    # Normalize audio data between -1.0 and 1.0 for processing
    audio_data = (audio_data - np.min(audio_data)) / (np.max(audio_data) - np.min(audio_data))
    audio_data = 2.0 * audio_data - 1.0

    # Process the audio using pedalboard
    fs = 44100  # Sample rate (44.1 kHz)
    filename = os.path.basename(image_path)
    frame_no = int(filename.replace("frame_", "").replace(".tiff", ""))
    processed_audio = effect(audio_data, fs, frame_no, frame_count)

    # Step 4: Convert processed audio back to an image
    # De-normalize audio data to original image range (0-255)
    processed_audio = ((processed_audio + 1.0) * 0.5 * 255).astype(np.uint8)
    processed_image_data = processed_audio.reshape(image_data.shape)

    # Step 5: Save the processed data as an image
    processed_image = Image.fromarray(processed_image_data)
    processed_image.save(output_path)
    image.close()

def process_frame_wrapper(args):
    """Helper wrapper since ProcessPoolExecutor needs a single argument."""
    return process_frame(*args)

def chunk_list(lst, chunk_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def main(filename):
    if not shutil.which("ffmpeg.exe"):
        raise EnvironmentError("ffmpeg is not installed or not found in system PATH.")

    input_path = os.path.abspath(filename)
    dry_output_path = os.path.join(os.getcwd(), "dry_output")
    wet_output_path = os.path.join(os.getcwd(), "wet_output")
    output_file_path = os.path.join(os.getcwd(), "output_" + filename)
    fps = 24
    max_workers = 4

    if not os.path.isdir(dry_output_path):
        extract_frames(input_path, dry_output_path, fps)

    os.makedirs(wet_output_path, exist_ok=True)

    # Prepare arguments for processing
    frame_paths = [(os.path.join(dry_output_path, filename), wet_output_path, len((os.listdir(dry_output_path))))
                   for filename in os.listdir(dry_output_path)]

    # Use concurrent.futures to process frames in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(process_frame_wrapper, frame_paths),
                  desc="Applying effects to frames", total=len(frame_paths)))

    reconstruct_video(wet_output_path, output_file_path, fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a video file')
    parser.add_argument('filename', type=str, help='The name of the video file to process')
    
    args = parser.parse_args()
    
    main(args.filename)