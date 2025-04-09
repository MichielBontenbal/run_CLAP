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
import paho.mqtt.client as mqtt
import json
#local imports
import config

# Global variables for thread communication
audio_queue = queue.Queue()
recording_active = threading.Event()

#variables for MQTT
mqtt_port = 31090
mqtt_host = config.mqtt_host
mqtt_user = config.mqtt_user
mqtt_password = config.mqtt_password 
app_id = "urban_sounds_clap"
dev_id = 'OE-007'
topic = "pipeline/urban_sounds_clap/OE-007"
client = mqtt.Client() # solving broken pipe issue
client.username_pw_set(mqtt_user, mqtt_password)

# FUNCTIONS 
def set_start():
    """ Set the start time of the recording """
    global start_time
    global unix_time
    start_time = datetime.datetime.now()
    unix_time = int(time.mktime(start_time.timetuple()))
    return start_time, unix_time

def get_cputemp():  # Code works on Raspberry Pi, exception when run on other platforms 
    """Get the CPU temperature of the Raspberry Pi"""
    try:
        temp = check_output(["vcgencmd", "measure_temp"]).decode("UTF-8")
        return float(findall("\d+\.\d+", temp)[0])
    except FileNotFoundError:
        return None  

def record_audio(duration=10, output_folder="samples"):
    """Record 10 s of audio and save it as a .wav file. Filename is starttime"""
    # Set the filename based on start time
    WAVE_OUTPUT_FILENAME = start_time.strftime("%Y-%m-%d_%H-%M-%S") + ".wav"

    # Audio recording constants
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 48000  # sample rate
    rate = 48000

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open stream
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=rate, input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")

    frames = []

    # Record for the specified duration
    for _ in range(0, int(rate / CHUNK * duration)):
        data = stream.read(CHUNK)
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
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

    print(f"Audio sample saved as {WAVE_OUTPUT_FILENAME}")
    
    return WAVE_OUTPUT_FILENAME

def generate_labels_list():
    """ Generate a list of candidate labels"""
    labels_list =['Gunshot', 'Alarm', 'Moped', 'Car', 'Motorcycle', 'Airplane', 'Helicopter', 'Claxon', 'Slamming door', 'Screaming', 'Talking','Music', 'Birds', 'Airco', 'Noise', 'Silence']
    return labels_list

def audio_classification(wav_file_path, labels_list):
    """ Classify an audio file based on a list of candidate labels using a zero-shot audio classification model."""
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
    """ Load audio with librosa"""
    # Load the sample with librosa
    y, sr = librosa.load(wav_file_path)
    return y, sr

def calculate_ptp(y):
    """ Calculate the peak-to-peak value of the audio sample"""
    ptp_value = float(np.ptp(y))
    print(f"Peak-to-peak value: {round(ptp_value, 4)}")
    return ptp_value

def create_spectrogram(wav_file_path, result, ptp_value):
    """ Generate a spectrogram of the audio sample with a caption showing the classification results and peak-to-peak value"""
    
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
    # plt.show() 
    # Save the plot
    spectogram_filename =  os.path.join("samples", start_time.strftime("%Y-%m-%d_%H-%M-%S") + ".png")
    plt.savefig(spectogram_filename, transparent=False, dpi=80, bbox_inches="tight")
    plt.close()
    return spec

def recording_thread():
    """Thread function for continuous audio recording"""
    while recording_active.is_set():
        try:
            #RATE = 48000
            #print(f"Using sampling rate: {RATE}")
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
                try:
                    result = audio_classification(wav_file_path, labels_list)
                except Exception as e:
                    print(f"Error during audio classification: {e}")
                    continue
                print(f"First result is {result[0]['label']}: {round(result[0]['score'],5)}")
                print(f"Second result is {result[1]['label']}: {round(result[1]['score'],5)}")
                print(f"Third result is {result[2]['label']}: {round(result[2]['score'],5)}")

                # Get CPU temperature
                RPI_temp = get_cputemp()
                print(f"RPi temperature: {RPI_temp}")
                
                # Analyze audio
                y, sr = load_audio_sample(wav_file_path)
                ptp_value = calculate_ptp(y)
                spec = create_spectrogram(wav_file_path, result, ptp_value)
                
                #Create a dictionary with top 5 results 
                mqtt_dict = {
                    result[0]['label']:result[0]['score'],
                    result[1]['label']:result[1]['score'],
                    result[2]['label']:result[2]['score'],
                    result[3]['label']:result[3]['score'],
                    result[4]['label']:result[4]['score']}
                mqtt_dict['start_recording']=unix_time
                mqtt_dict['RPI_temp']=RPI_temp 
                mqtt_dict['ptp']=ptp_value
                mqtt_dict['spectrogram']=str(spec)
                
                # Create the MQTT message and convert to JSON 
                mqtt_message = {
                    "app_id": app_id,
                    "dev_id": dev_id, 
                    "payload_fields": mqtt_dict,
                    "time": time.time()*1000
                    }
                msg_str = json.dumps(mqtt_message)
                #print(topic)
                print(msg_str)
            
                # Connect to  MQTT client 
                try:
                    client.connect(mqtt_host)
                except paho.mqtt.client.MQTTException as e:
                    print(f"MQTT connection error: {e}")

                # Publish the message
                try:
                    client.publish(topic, msg_str)
                    print('message sent')
                except Exception as e:
                    print(f"A connection error occurred: {e}")
                finally:
                    client.disconnect()		

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
