#!/usr/bin/env python
# coding: utf-8
print('Starting script...')

#import standard python packages
import numpy as np
from matplotlib import pyplot as plt
import datetime
import time
import os
import threading
import queue
#import audio packages
import soundfile as sf
import librosa
import librosa.display
import pyaudio
import wave 
#imports for deep learning
from transformers import pipeline
#imports for cpu temp 
from subprocess import check_output
from re import findall
#imports for mqtt
import paho.mqtt.publish as publish
import urban_sounds_config
import json

# Global variables for thread communication
audio_queue = queue.Queue()
recording_active = threading.Event()

#variables for MQTT
mqtt_host = 
mqtt_port = 1883
mqtt_user = 
mqtt_password = 

# FUNCTIONS 
def set_start():
    global start_time
    """Set the start time of the recording"""
    start_time = datetime.datetime.now()
    return start_time

def get_cputemp():
    temp = check_output(["vcgencmd","measure_temp"]).decode("UTF-8")
    return float(findall("\d+\.\d+",temp)[0])

def record_audio(duration=10, output_folder="samples"):
    """Record 10 s of audio and save it as a .wav file. Filename is starttime"""
    # Set the filename based on start time
    
    WAVE_OUTPUT_FILENAME = start_time.strftime("%Y-%m-%d_%H-%M-%S") + ".wav"

    # Audio recording parameters
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    rate = 48000

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open stream
    stream = audio.open(format=format, channels=channels,
                        rate=rate, input=True,
                        frames_per_buffer=chunk)

    print("Recording...")

    frames = []

    # Record for the specified duration
    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Recording finished.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define the output filename with the output folder
    WAVE_OUTPUT_FILENAME = os.path.join(output_folder, WAVE_OUTPUT_FILENAME)

    # Save the recorded data as a WAV file
    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

    print(f"Audio sample saved as {WAVE_OUTPUT_FILENAME}")
    
    return WAVE_OUTPUT_FILENAME

def generate_labels_list():
    """Generate a list of candidate labels"""
    labels_list =['Gunshot', 'Alarm', 'Moped', 'Car', 'Motorcycle', 'Claxon', 'Slamming door', 'Screaming', 'Talking','Music', 'Birds', 'Airco', 'Noise', 'Silence']
    return labels_list

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

def load_audio_sample(wav_file_path):
    """ load audio with librosa"""
    # Load the sample with librosa
    y, sr = librosa.load(wav_file_path)
    return y, sr

def calculate_ptp(y):
    """Calculate the peak-to-peak value of the audio sample"""
    ptp_value = float(np.ptp(y))
    print(f"Peak-to-peak value: {round(ptp_value, 4)}")
    return ptp_value

def create_spectrogram(wav_file_path, result, ptp_value):
    """Generate a spectrogram of the audio sample with a caption showing the classification results and peak-to-peak value"""
    
    y, sr = librosa.load(wav_file_path)
    spec = np.abs(librosa.stft(y, hop_length=512))
    spec = librosa.amplitude_to_db(spec, ref=np.max)

    plt.figure()
    librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectogram {wav_file_path}', fontsize=10)

    # Caption with multiple lines
    caption = (f"{result[0]['label']}: {round(result[0]['score'], 4)} - "
               f"{result[1]['label']}: {round(result[1]['score'], 4)} - "
               f"{result[2]['label']}: {round(result[2]['score'], 4)} - "
               f'p2p: {round(ptp_value, 4)}')
    plt.figtext(0.5, 0.01, caption, ha="center", fontsize=10)
    plt.xlabel('')
    #plt.show() 
    #save the plot
    spectogram_filename =  os.path.join("samples", start_time.strftime("%Y-%m-%d_%H-%M-%S") + ".png")
    plt.savefig(spectogram_filename, transparent=False, dpi=80, bbox_inches="tight")

def recording_thread():
    """Thread function for continuous audio recording"""
    while recording_active.is_set():
        try:
            start_time = set_start()
            wav_file_path = record_audio()
            audio_queue.put((start_time, wav_file_path))
        except Exception as e:
            print(f"Error in recording thread: {e}")

def processing_thread():
    """Thread function for audio classification and analysis"""
    while recording_active.is_set(): #
        try:
            if not audio_queue.empty():
                start_time, wav_file_path = audio_queue.get()
                
                # Generate labels and classify
                labels_list = generate_labels_list()
                result = audio_classification(wav_file_path, labels_list)
                print(f"First result is {result[0]['label']}: {round(result[0]['score'],5)}")
                print(f"Second result is {result[1]['label']}: {round(result[1]['score'],5)}")
                print(f"Third result is {result[2]['label']}: {round(result[2]['score'],5)}")

                # Get CPU temperature
                cpu_temp = get_cputemp()
                print(f"CPU Temperature: {cpu_temp}")
                
                # Analyze audio
                y, sr = load_audio_sample(wav_file_path)
                ptp_value = calculate_ptp(y)
                create_spectrogram(wav_file_path, result, ptp_value)
                
                #Creating a dict with top 3 
                mqtt_dict = {
                    result[0]['label']:result[0]['score'],
                    result[1]['label']:result[1]['score'],
                    result[2]['label']:result[2]['score'],
                    result[3]['label']:result[3]['score'],
                    result[4]['label']:result[4]['score']}
                mqtt_dict['cputemp']=cpu_temp 
                mqtt_dict['ptp_value']=ptp_value

                #creating the mqtt message 
                msg_json = { 
                "app_id": 'michiel',
                "dev_id": "OE003",
                "payload_fields": mqtt_dict,
                "time": int(time.time() * 1e3)
                }
                msg_str = json.dumps(msg_json)
                print(msg_str)
                try: 
                    auth = {"username": mqtt_user, "password": mqtt_password}
                    publish.single("pipeline/openears/OE001", payload=msg_str, hostname=mqtt_host, port=mqtt_port, auth=auth)
                    print('message sent')
                except:
                    print('a mqtt connection error occured')

                audio_queue.task_done()
        except Exception as e:
            print(f"Error in processing thread: {e}")

def main():
    try:
        # Create and start threads
        recording_active.set()
        recorder = threading.Thread(target=recording_thread)
        processor = threading.Thread(target=processing_thread)
        
        recorder.start()
        processor.start()
        
        processor.join()

        # Keep the main thread running
        while True:
            pass
            
    except KeyboardInterrupt:
        print("\nStopping threads...")
        recording_active.clear()
        recorder.join()
        processor.join()
        print("Threads stopped successfully")

if __name__ == "__main__":
    main()
