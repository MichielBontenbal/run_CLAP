import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from transformers import pipeline
import datetime

SAVE_RECORDING = True

def set_start():
    """ Set the start time of the recording """
    global start_time
    start_time = datetime.datetime.now()
    return start_time

def record_audio(duration=10, sample_rate=48000, save_to_file=False):
    """ Records audio for a specified duration and optionally saves it as a .wav file."""
    
    WAVE_OUTPUT_FILENAME = start_time.strftime("%Y-%m-%d_%H-%M-%S") + ".wav"

    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")

    # Save to .wav file if save_to_file is True
    if save_to_file:
        wav.write(WAVE_OUTPUT_FILENAME, sample_rate, audio_data)
        print(f"Audio saved to {WAVE_OUTPUT_FILENAME}")

    return sample_rate, audio_data.flatten()

def generate_labels_list():
    """ Generate a list of candidate labels"""
    labels_list =['Gunshot', 'Alarm', 'Moped', 'Car', 'Motorcycle', 'Airplane', 'Helicopter', 'Claxon', 'Slamming door', 'Screaming', 'Talking','Music', 'Birds', 'Airco', 'Noise', 'Silence']
    return labels_list

def audio_classification(audio_data, labels_list):
    """ Classify an audio file based on a list of candidate labels using a zero-shot audio classification model."""
    try:
        # Read the audio file
        #audio, samplerate = sf.read(wav_file_path)

        # Initialize the audio classifier pipeline
        audio_classifier = pipeline(task="zero-shot-audio-classification", model="laion/larger_clap_general")

        # Perform classification
        output = audio_classifier(audio_data, candidate_labels=labels_list)

        return output

    except FileNotFoundError:
        return "Error: The specified audio file was not found."
    except Exception as e:
        return f"An error occurred: {e}"
    
# Example usage:
set_start()
sample_rate, audio_data = record_audio(duration=5, save_to_file=SAVE_RECORDING)
labels_list = generate_labels_list()
result = audio_classification(audio_data, labels_list)
print(result)

