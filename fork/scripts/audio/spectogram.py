"""
SPDX-License-Identifier: MIT
Copyright © 2026 dragos2001
"""

import torchaudio
import torch
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
from typing import BinaryIO
import cv2

SR = 22050                # target sample rate
WIN_SIZE = 512           # Hann window size
HOP_SIZE = 128           # hop length
FFT_SIZE = 1024          # FFT length
N_MELS = 128             # mel bins
FMIN = 50                # min frequency
FMAX = SR // 2           # max frequency (11.025 kHz)
WINDOW_SEC = 3.0         # window length in seconds
STRIDE_SEC = 1.5         # stride (overlap)
TARGET_IMG_SIZE = (224, 224)  # final image size


def info_audio(audio_stream: BytesIO):
    waveform, sample_rate = torchaudio.load(audio_stream)
    duration = waveform.size(1) / sample_rate
    info = {
        "sample_rate": sample_rate,
        "duration": duration,
        "num_channels": waveform.size(0)
    }

    return waveform, sample_rate, info 

def resample(audio_tensor: torch.Tensor, original_sr: int, target_sr: int) -> torch.Tensor:
    if original_sr == target_sr:
        return audio_tensor
    resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
    return resampler(audio_tensor)

def partition_audio_stream(audio_stream: BytesIO, chunk_length_seconds: int = 3):
    waveform, sample_rate, info  = info_audio(audio_stream)
    print(f"Partitioning audio stream with info: {info}")

    #resample if neccessary
    if sample_rate != 22050:
        waveform = resample(waveform, sample_rate, 22050)
        sample_rate = 22050

    #extract signal length
    total_length = waveform.size(1)

    #calculate chunk length in samples given chunk length in seconds in seconds
    chunk_length = chunk_length_seconds * sample_rate
    print(f"Chunk length in samples: {chunk_length}")
    #list to hold audio chunks
    chunks = []

    #iterate over the waveform to create overlapping chunks
    for start in range(0, int(total_length), chunk_length//2 + 1):
        end = start + chunk_length
        if end > total_length:
            chunk = waveform[:, start:total_length]
            if chunk.shape[1] < 1+ (end - total_length):
                break
            
            print(f"Padding chunk from of {chunk.shape} to {end - start} samples")
            torch.nn.functional.pad(chunk, (0, end - total_length),mode='reflect')
        else:
            chunk = waveform[:, start:end]
            print(f"Created chunk of chunk size {chunk.shape } samples")
        
        chunks.append(chunk)
        
    return chunks

def save_spectogram_image(spectrogram: np.ndarray, output_path: str):
    #save to spectogram path
    np.save(output_path, spectrogram)
   

def generate_spectogram(audio_tensor: torch.Tensor, stack: bool = False) -> Image.Image:
    #spectogram transform
    spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR,
        n_mels=N_MELS,
        n_fft=FFT_SIZE,
        hop_length=HOP_SIZE
    )(audio_tensor)

    #spectogram in db
    spectrogram_db = torchaudio.transforms.AmplitudeToDB()(spectrogram)
    
    #mel normalized
    mel_norm = 255 * (spectrogram_db - torch.min(spectrogram_db)) / (torch.max(spectrogram_db) - torch.min(spectrogram_db))
    mel_norm = mel_norm.numpy().astype(np.uint8)
    mel_norm = np.squeeze(mel_norm)
    
    #nek resized
    mel_resized = cv2.resize(mel_norm, (224, 224), interpolation=cv2.INTER_LINEAR)
    mel_resized = cv2.resize(mel_resized, TARGET_IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    
    #check for stack
    if stack:
        mel_resized = np.stack([mel_resized]*3, axis=0)

    return mel_resized