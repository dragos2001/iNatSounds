
import os 
import sys
from multiprocessing import Pool, cpu_count
from io import BytesIO
from typing import BinaryIO
import argparse
import torchaudio
from .spectogram import generate_spectogram, partition_audio_stream,save_spectogram_image

def process_file(file_path, spectrogram_output_dir):
     
        print(f"Processing species: {os.path.basename(os.path.dirname(file_path))}, file: {os.path.basename(file_path)}")

        with open(file_path, 'rb') as audio_file:

            #audio stream 
            audio_stream = BytesIO(audio_file.read())
            directory = os.path.basename(os.path.dirname(file_path))
            filename = os.path.basename(file_path)
    

            # Partition audio into chunks
            audio_chunks = partition_audio_stream(audio_stream, chunk_length_seconds=3)
            for idx, chunk in enumerate(audio_chunks):
                spectrogram_path = os.path.join(spectrogram_output_dir, directory, f"{os.path.splitext(filename)[0]}_spectrogram_chunk_{idx}.npy")
                if os.path.exists(spectrogram_path):
                    print(f"Spectrogram already exists at: {spectrogram_path}, skipping...")
                    continue
                spectrogram = generate_spectogram(chunk)
                print(f"Generated spectrogram for chunk {idx} of file {filename} has size: {spectrogram.shape}")
                spectrogram_dir = os.path.join(spectrogram_output_dir, directory)
                os.makedirs(spectrogram_dir, exist_ok=True)
                save_spectogram_image(spectrogram, spectrogram_path)
                print(f"Saved spectrogram to: {spectrogram_path}")
            #audio_chunks = partition_audio_stream(audio_stream, chunk_length=3)  
            
        return True
def main():

  
    #spectogram directory
    spectogram_dir = "/mnt/tmp_spectograms"
    
    #wav directory
    wav_dir = "/mnt/tmp_audio/train/train"
    


    #spectogram output directory
    spectograms_output_dir = os.path.join(spectogram_dir, os.path.basename(wav_dir))
    os.makedirs(spectograms_output_dir, exist_ok=True)


    #wav output directory
    for directory in os.listdir(wav_dir):
            dir_path = os.path.join(wav_dir, directory)
            if os.path.isdir(dir_path):
                print(f"Processing directory: {dir_path}")
                file_paths = [os.path.join(dir_path, filename) for filename in os.listdir(dir_path) if filename.lower().endswith((".wav", ".flac", ".mp3"))]
                print(f"Found {len(file_paths)} audio files in directory {dir_path}")   
                with Pool(cpu_count()) as pool:
                    results = pool.starmap(process_file, [(file_path, spectograms_output_dir) for file_path in file_paths])
                    

if __name__ == "__main__":
    main()