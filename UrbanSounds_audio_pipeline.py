#!/usr/bin/env python
# coding: utf-8

#import basic python packages
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
#import audio packages
import soundfile as sf
import librosa
import librosa.display
import pyaudio
import wave 
#import deep learning packages
import torch #not used but necessary for transformers
from transformers import pipeline


# 1. TAKE AUDIO SAMPLE AND SAVE IT

#set the filename based on current time
current_time = datetime.datetime.now()
WAVE_OUTPUT_FILENAME = current_time.strftime("%Y-%m-%d_%H-%M-%S") + ".wav"

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
RECORD_SECONDS = 10

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open stream
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("Recording...")

frames = []

# Record for RECORD_SECONDS
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Recording finished.")

# Stop and close the stream
stream.stop_stream()
stream.close()
audio.terminate()

# Save the recorded data as a WAV file

# Ensure the "samples" folder exists
os.makedirs("samples", exist_ok=True)

# Define the output filename with the "samples" folder
WAVE_OUTPUT_FILENAME = os.path.join("samples", WAVE_OUTPUT_FILENAME)

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
print(f"Audio sample saved as {WAVE_OUTPUT_FILENAME}")

# 2. CLASSIFY AUDIO SAMPLE 

wav_file_path = WAVE_OUTPUT_FILENAME

# With the similarity search approach of CLAP we can now use a whole range of labels 
labels_list =['Gunshot', 'Alarm', 'Moped', 'Car', 'Motorcycle', 'Claxon', 'Slamming door', 'Screaming', 'Talking','Music', 'Birds', 'Airco', 'Noise', 'Silence']

def audio_classification(wav_file_path, labels_list):
    """
    Classify an audio file based on a list of candidate labels using a zero-shot audio classification model.
    """
    try:
        # Read the audio file
        audio, samplerate = sf.read(wav_file_path)

        # Initialize the audio classifier pipeline
        audio_classifier = pipeline(task="zero-shot-audio-classification", model="laion/larger_clap_general")

        # Perform classification
        output = audio_classifier(audio, candidate_labels=labels_list)

        return output

    except FileNotFoundError:
        return "Error: The specified audio file was not found."
    except Exception as e:
        return f"An error occurred: {e}"

#call the function
result = audio_classification(wav_file_path, labels_list)

print(f'First result is {result[0]['label']}: {result[0]['score']}')
print(f'Second result is {result[1]['label']}: {result[1]['score']}')
print(f'Third result is {result[2]['label']}: {result[2]['score']}')


# 3. DATA ANALYSIS

#load the sample with librosa and calculate Peak-2-peak
y, sr = librosa.load(wav_file_path)

ptp_value = np.ptp(y)
print(f"Peak-to-peak value: {ptp_value}")

#create a spectogram
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # or "true"
spec = np.abs(librosa.stft(y, hop_length=512))
spec = librosa.amplitude_to_db(spec, ref=np.max)

plt.figure()
#librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
spectogram_filename = current_time.strftime("%Y-%m-%d_%H-%M-%S") + ".png"
plt.savefig(spectogram_filename, transparent=False, dpi=80, bbox_inches="tight")
#plt.show()



