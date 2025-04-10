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
import sounddevice as sd
import scipy.io.wavfile as wav
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

# Setting to save recording of audio 
SAVE_RECORDING = True

#Settings for MQTT
mqtt_port = 31090
mqtt_host = config.mqtt_host
mqtt_user = config.mqtt_user
mqtt_password = config.mqtt_password 
app_id = "urbansounds"
dev_id = 'OE-007'
topic = "pipeline/urbansounds/OE-007"
client = mqtt.Client() # solving broken pipe issue
client.username_pw_set(mqtt_user, mqtt_password)

# Global variables for thread communication
audio_queue = queue.Queue()
recording_active = threading.Event()

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

def record_audio(duration=10, output_folder="samples", save_to_file=False):

    # Audio recording constants
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    SAMPLE_RATE = 48000  # sample rate
    sample_rate= SAMPLE_RATE
    """ Records audio for a specified duration and optionally saves it as a .wav file."""
    
    WAVE_OUTPUT_FILENAME = start_time.strftime("%Y-%m-%d_%H-%M-%S") + ".wav"

    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define the output filename with the output folder
    WAVE_OUTPUT_FILENAME = os.path.join(output_folder, WAVE_OUTPUT_FILENAME)

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
    

def calculate_ptp(audio_data):
    """ Calculate the peak-to-peak value of the audio data """
    return np.ptp(audio_data)

def create_spectrogram(audio_data, sample_rate):
    """ Create and display a spectrogram from audio data """
    # Compute the Short-Time Fourier Transform (STFT)
    stft = librosa.stft(audio_data)
    # Convert the complex-valued STFT to magnitude
    spectrogram_data = np.abs(stft)
    return spectrogram_data

    # Display the spectrogram
    #plt.figure(figsize=(10, 6))
    #librosa.display.specshow(librosa.amplitude_to_db(spectrogram_data, ref=np.max),
    #                         sr=sample_rate, x_axis='time', y_axis='log')
    #plt.colorbar(format='%+2.0f dB')
    #plt.title('Spectrogram')
    #plt.show()

def recording_thread():
    """Thread function for continuous audio recording"""
    while recording_active.is_set():
        try:
            #RATE = 48000
            #print(f"Using sampling rate: {RATE}")
            start_time = set_start()
            sample_rate, audio_data = record_audio(duration=5, save_to_file=SAVE_RECORDING)
            #wav_file_path = record_audio()
            audio_queue.put((start_time, audio_data))
        except Exception as e:
            print(f"Error in recording thread: {e}")

def processing_thread():
    """Thread function for audio classification and analysis"""
    while recording_active.is_set(): #
        try:
            if not audio_queue.empty():
                start_time, audio_data = audio_queue.get()
                print('start classifying:')
                # Generate labels and classify
                labels_list = generate_labels_list()
                try:
                    result = audio_classification(audio_data, labels_list)
                except Exception as e:
                    print(f"Error during audio classification: {e}")
                    continue
                print(f"First result is {result[0]['label']}: {round(result[0]['score'],5)}")
                print(f"Second result is {result[1]['label']}: {round(result[1]['score'],5)}")
                print(f"Third result is {result[2]['label']}: {round(result[2]['score'],5)}")

                # Get CPU temperature
                RPI_temp = get_cputemp()
                print(f"RPi temperature: {RPI_temp}")

                # Analyse audio
                ptp_value = calculate_ptp(audio_data)
                sample_rate = 48000
                spectrogram_data = create_spectrogram(audio_data, sample_rate)
                #print(f'spectrogram_data: {spectrogram_data}')

                print('making mqtt_dict')
                #Create a dictionary with top 5 results 
                mqtt_dict = {
                    result[0]['label']:result[0]['score'],
                    result[1]['label']:result[1]['score'],
                    result[2]['label']:result[2]['score'],
                    result[3]['label']:result[3]['score'],
                    result[4]['label']:result[4]['score']}
                #mqtt_dict['start_recording']=start_time-10 #lelijke oplossing om de start tijd goed te krijgen. Moet nog beter
                mqtt_dict['RPI_temp']=RPI_temp 
                mqtt_dict['ptp']=ptp_value
                mqtt_dict['spectrogram']=spectrogram_data.tolist()
                #print(f'mqtt_dict: {mqtt_dict}')
              
                # Convert all float32 values in mqtt_dict to native Python float
                mqtt_dict = {key: float(value) if isinstance(value, np.float32) else value for key, value in mqtt_dict.items()}

                # Now you can safely serialize mqtt_dict to JSON
                mqtt_json = json.dumps(mqtt_dict)

                # Create the MQTT message and convert to JSON 
                mqtt_message = {
                    "app_id": app_id,
                    "dev_id": dev_id, 
                    "payload_fields": mqtt_dict,
                    "time": int(time.time()*1000)
                    }
                msg_str = json.dumps(mqtt_message)
                print(topic)
                #print(msg_str)
            
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
