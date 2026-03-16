#!/usr/bin/env python
"""
Test scripts to calibrate for RMS and dB SPL.
"""

import librosa
import pyaudio
import sounddevice as sd
import datetime
import os
import scipy.io.wavfile as wav
import numpy as np


# Some constants
DURATION = 5  # duration of each audio recording in seconds 
SAVE_RECORDING = False # whether to save the recorded audio as .wav files   
OFFSET = 0.0  # offset for dB SPL calculation (to be calibrated based on the microphone sensitivity and recording setup)

def set_start():
    """Set the start time of the recording"""
    start_time = datetime.datetime.now()
    return start_time

def record_audio(duration, output_folder="samples", save_to_file=False, start_time=None):
    """Records audio for a specified duration and optionally saves it as a .wav file."""
    # Audio recording constants
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    SAMPLE_RATE = 48000  # sample rate
    sample_rate = SAMPLE_RATE

    WAVE_OUTPUT_FILENAME = start_time.strftime("%Y-%m-%d_%H-%M-%S") + ".wav"

    try:
        # print("Recording...")
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
        )
        sd.wait()  # Wait until recording is finished
    except Exception as e:
        print(f"Error during audio recording: {e}")
        return None, None

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define the output filename with the output folder
    WAVE_OUTPUT_FILENAME = os.path.join(output_folder, WAVE_OUTPUT_FILENAME)

    # Save to .wav file if save_to_file is True
    if save_to_file:
        wav.write(WAVE_OUTPUT_FILENAME, sample_rate, audio_data)
        print(f"Audio saved to {WAVE_OUTPUT_FILENAME}")

    return sample_rate, audio_data.flatten()


def create_rms(audio_data):
    """Create a RMS value from audio data and convert to data"""
    rms = librosa.feature.rms(y=audio_data)
    return rms


def calculate_db_spl(rms):
    """Calculate dB SPL from RMS values using the formula: db_spl = 20 * np.log10(rms) + OFFSET"""
    db_spl = 20 * np.log10(rms) + OFFSET
    return db_spl

start_time = set_start()
sample_rate, audio_data = record_audio(DURATION, start_time=start_time)
rms = create_rms(audio_data)
db_spl = calculate_db_spl(rms)
print(f"RMS: {rms.flatten()[0]:.6f}, dB SPL: {db_spl.flatten()[0]:.2f} dB")