#!/usr/bin/env python3

import subprocess
import os
import shutil
from pedalboard import Pedalboard, Phaser, Delay, Chorus, Limiter, Gain, Compressor
from noisereduce import reduce_noise
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

FILE_EXT = '.tiff'
SAMPLE_RATE = 44100

def reduce(audio_data):
    noise_signal = audio_data[:100]
    reduction = reduce_noise(y=audio_data, 
                            sr=SAMPLE_RATE, 
                            freq_mask_smooth_hz=87,
                            time_mask_smooth_ms=6,
                            thresh_n_mult_nonstationary=0.1,
                            sigmoid_slope_nonstationary=50,
                            n_fft=1024)
    residue = audio_data - reduction
    return reduction

def effect(audio_data, fs, frame_no, frame_count):
    coef = float(frame_no) / float(frame_count)
    audio_data = audio_data[::-1]
    board = Pedalboard([
        Delay(delay_seconds=0.005, feedback=0.8, mix=1),
        #Chorus()
        #Gain(gain_db = 3),
        Phaser(rate_hz=0.005, feedback=0.5, mix=1),
        Limiter(threshold_db=-6),
    ])
    audio_data = board(audio_data, fs)
    audio_data = reduce(audio_data)
    audio_data = audio_data[::-1]

    return audio_data

def extract_frames(file_path, output_dir, fps):
    os.makedirs(output_dir, exist_ok=True)
    ffmpeg_command = [
        'ffmpeg',
        '-i', file_path,
        '-vf', f"fps={fps}",
        os.path.join(output_dir, 'frame_%06d' + FILE_EXT)
    ]
    subprocess.run(ffmpeg_command, check=True)
    print(f"Extracted {len(os.listdir(output_dir))} frames at {fps} fps")

def reconstruct_video(image_directory, output_path, fps):
    if not os.path.exists(image_directory):
        raise ValueError("Image directory does not exist.")

    command = [
        'ffmpeg',
        '-y', # automatically overwrite file
        '-framerate', str(fps),
        '-i', os.path.join(image_directory, 'frame_%06d.png'), 
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

def process_frame_stereo(image_path, output_dir, frame_count):
    base_name, _ = os.path.splitext(os.path.basename(image_path))
    output_path = os.path.join(output_dir, f"{base_name}.png")

    image = Image.open(image_path)
    image_data = np.array(image)

    # Flatten image data to 1D
    audio_data = image_data.flatten().astype(np.float32)

    # Normalize audio data between -1.0 and 1.0
    audio_data = (audio_data - np.min(audio_data)) / (np.max(audio_data) - np.min(audio_data))
    audio_data = 2.0 * audio_data - 1.0

    # Convert to stereo (2-channel)
    stereo_audio_data = np.vstack((audio_data, audio_data)).T

    # Process the audio using the pedalboard equivalent (dummy processing here)
    fs = SAMPLE_RATE  # Sample rate (44.1 kHz)
    filename = os.path.basename(image_path)
    frame_no = int(filename.replace("frame_", "").replace(FILE_EXT, ""))  # Assuming FILE_EXT is '.png'
    processed_audio = effect(stereo_audio_data.flatten(), fs, frame_no, frame_count)

    # Convert the processed audio back to mono by averaging
    processed_audio_mono = processed_audio.reshape(-1, 2).mean(axis=1)

    # De-normalize audio data to original image range (0-255)
    processed_audio_mono = ((processed_audio_mono + 1.0) * 0.5 * 255).astype(np.uint8)
    processed_image_data = processed_audio_mono.reshape(image_data.shape)
    
    # Save the processed data as an image
    processed_image = Image.fromarray(processed_image_data)
    processed_image.save(output_path)
    image.close()

def process_frame(image_path, output_dir, frame_count):
    base_name, _ = os.path.splitext(os.path.basename(image_path))
    output_path = os.path.join(output_dir, f"{base_name}.png")
    # if (os.path.isfile(output_path)): return

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
    frame_no = int(filename.replace("frame_", "").replace(FILE_EXT, ""))
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
    return process_frame_stereo(*args)

def chunk_list(lst, chunk_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def main(filename):
    if not shutil.which("ffmpeg"):
        raise EnvironmentError("ffmpeg is not installed or not found in system PATH.")
    
    fps = 24
    max_workers = os.cpu_count() # alternatively, os.cpu_count()
    batch_size = 10  # Adjust this based on your memory constraints
    start_frame = 0
    max_frames = 500

    input_path = os.path.abspath(filename)
    dry_output_path = os.path.join(os.getcwd(), "dry_output")
    wet_output_path = os.path.join(os.getcwd(), "wet_output")
    output_file_path = os.path.join(os.getcwd(), "output_" + filename)

    if not os.path.isdir(dry_output_path):
        extract_frames(input_path, dry_output_path, fps)

    os.makedirs(wet_output_path, exist_ok=True)

    # Prepare arguments for processing
    frame_paths = [(os.path.join(dry_output_path, filename), wet_output_path, len((os.listdir(dry_output_path))))
                   for filename in sorted(os.listdir(dry_output_path))][start_frame:max_frames]

    # Process frames in batches
    with tqdm(total=len(frame_paths), desc="Applying effects to frames") as pbar:
        for batch_frame_paths in chunk_list(frame_paths, batch_size):
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for _ in executor.map(process_frame_wrapper, batch_frame_paths):
                    pbar.update()

    reconstruct_video(wet_output_path, output_file_path, fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a video file')
    parser.add_argument('filename', type=str, help='The name of the video file to process')
    
    args = parser.parse_args()
    
    main(args.filename)