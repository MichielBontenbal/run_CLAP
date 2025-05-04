#!/usr/bin/env python
# coding: utf-8
print("Starting script...")

# import standard python packages
import datetime
import gc  # garbage collection
from matplotlib import pyplot as plt
import numpy as np
import os
import queue
import threading
import time

# import audio packages
import sounddevice as sd
import scipy.io.wavfile as wav
import soundfile as sf
import librosa
import pyaudio

# imports for cpu temp
from subprocess import check_output
from re import findall

# imports for mqtt
import paho.mqtt.client as mqtt
from paho.mqtt.client import CallbackAPIVersion 
import json

# local imports
import config
import sound_scapes # a file to load the labels

# Setting the Huggingface tokenizer setting and importing the pipeline
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # must be done before importing transformers)
from transformers import pipeline

# Setting to save recording of audio
SAVE_RECORDING = False

# Settings for and initialization for MQTT
mqtt_port = 31090
mqtt_host = config.mqtt_host
mqtt_user = config.mqtt_user
mqtt_password = config.mqtt_password
app_id = "urbansounds"
dev_id = "OE-007"
topic = "pipeline/urbansounds/OE-007"
# client = mqtt.Client()  # solving broken pipe issue
client = mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION2)
client.username_pw_set(mqtt_user, mqtt_password)

# Variables for thread communication
audio_queue = queue.Queue()
recording_active = threading.Event()
queue_lock = threading.Lock()  # Add a lock for the queue


# FUNCTIONS
def set_start():
    """Set the start time of the recording"""
    start_time = datetime.datetime.now()
    return start_time


def get_cputemp():
    """Get the CPU temperature of the Raspberry Pi, exception when run on other platforms"""
    try:
        temp = check_output(["vcgencmd", "measure_temp"]).decode("UTF-8")
        return float(findall("\d+\.\d+", temp)[0])
    except Exception as e:
        print(f"{e}Cannot get temperature, this code is intended for Raspberry Pi.")


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


def generate_labels_list():
    """Generate a list of candidate labels"""
    labels_list = sound_scapes.marineterrein_labels
    return labels_list


def initialize_audio_classifier():
    """Initialize the zero-shot audio classification model."""
    return pipeline(task="zero-shot-audio-classification", model="laion/larger_clap_general")


def audio_classification(audio_classifier, audio_data, labels_list):
    """Classify an audio file based on a list of candidate labels using the initialized audio classifier."""
    try:
        # Perform classification
        output = audio_classifier(audio_data, candidate_labels=labels_list)
        return output
    except Exception as e:
        print(f"An error occurred during classification: {e}")


def calculate_ptp(audio_data):
    """Calculate the peak-to-peak value of the audio data"""
    return np.ptp(audio_data)


def create_spectrogram(audio_data, sample_rate):
    """Create a spectrogram from audio data and convert to data"""
    stft = librosa.stft(audio_data)
    spectrogram_data = np.abs(stft)
    return spectrogram_data

def create_rms(audio_data):
    """Create a RMS value from audio data and convert to data"""
    rms = librosa.feature.rms(y=audio_data)
    return rms


def recording_thread():
    """Thread function for continuous audio recording"""
    while recording_active.is_set():
        print("start recording thread")
        try:
            start_time = set_start()
            sample_rate, audio_data = record_audio(
                duration=10, save_to_file=SAVE_RECORDING, start_time=start_time
            )
            with queue_lock:  # Acquire lock before putting into the queue
                audio_queue.put((start_time, audio_data))
        except Exception as e:
            print(f"Error in recording thread: {e}")
        print("recording thread completed")
        time.sleep(0.1)  # Add a small delay to reduce CPU usage / avoid busy-waiting


def processing_thread():
    """Thread for audio classification and sending mqtt message"""
    while recording_active.is_set():
        # print('start processing thread')
        time.sleep(0.1)  # Add a small delay to reduce CPU usage / avoid busy-waiting
        try:
            try:
                with queue_lock:  # Acquire lock before getting from the queue
                    start_time, audio_data = audio_queue.get(
                        timeout=1
                    )  # Add a timeout to avoid indefinite blocking
                    print(f"processing sample with start time: {start_time}")
            except queue.Empty:
                print("Queue is empty, waiting for audio data...")
                time.sleep(0.5)  # Wait for a short time before checking again
                continue

            # Classification
            print("start classifying:")

            try:
                result = audio_classification(audio_classifier, audio_data, labels_list)
            except Exception as e:
                print(f"Error during audio classification: {e}")

            print(f"""Classifications: {result[0]['label']}: {round(result[0]['score'],5)} | {result[1]['label']}: {round(result[1]['score'],5)} | {result[2]['label']}: {round(result[2]['score'],5)} | {result[3]['label']}: {round(result[3]['score'],5)} | {result[4]['label']}: {round(result[4]['score'],5)}""")
            total = result[0]['score'] + result[1]['score'] + result[2]['score'] + result[3]['score'] + result[4]['score']
            print(f"Total score: {total}")
            # Get CPU temperature
            RPI_temp = get_cputemp()

            # Analyse audio
            ptp_value = calculate_ptp(audio_data)
            sample_rate = 48000 # to clean up
            spectrogram_data = create_spectrogram(audio_data, sample_rate)

            # add the start_time to the mqtt message as unixtime
            unix_time = int(time.mktime(start_time.timetuple()))

            # Create a dictionary for the MQTT message
            mqtt_dict = {}
            for i, result_item in enumerate(result[:5]):  # Limit to top 5
                mqtt_dict[result_item["label"]] = result_item["score"]
            # Add additional data to the dictionary
            mqtt_dict["start_recording"] = unix_time
            mqtt_dict["RPI_temp"] = RPI_temp
            mqtt_dict["ptp"] = ptp_value
            #mqtt_dict["spectrogram"] = spectrogram_data.tolist()  # Maybe later in the project (when we'll start data analysis)
            #mqtt_dict["rms"] = create_rms(audio_data).tolist()  # Idem for later
            
            # Convert all float32 values in mqtt_dict to native Python float
            mqtt_dict = {
                key: float(value) if isinstance(value, np.float32) else value
                for key, value in mqtt_dict.items()
            }

            # Create the MQTT message and convert to JSON
            mqtt_message = {
                "app_id": app_id,
                "dev_id": dev_id,
                "payload_fields": mqtt_dict,
                "time": int(time.time() * 1000),
            }

            msg_str = json.dumps(mqtt_message)
            # print(msg_str)

            # Publish the message
            try:
                # Check if the client is connected
                if client.is_connected():
                    client.publish(topic, msg_str)
                    print("Connection up & MQTT message sent successfully")
                else:
                    client.reconnect()
                    client.publish(topic, msg_str)
                    print("MQTT message sent after reconnection")
            except Exception as e:
                print(f"Unexpected error while publishing MQTT message: {e}")

            # Clean the memory
            gc.collect()
            #print("garbage collected")
            
            audio_queue.task_done()
            time.sleep(0.5)  # Add a small delay to reduce CPU usage 

        except Exception as e:
            print(f"Error in processing thread: {e}")


def main():
    try:
        # Initialize the audio classifier and load the labels
        global audio_classifier, labels_list
        audio_classifier = initialize_audio_classifier()
        labels_list = generate_labels_list()

        # Connect to MQTT client
        try:
            client.connect(mqtt_host)
            client.loop_start()  # Start the MQTT network loop
            print("Connected to MQTT broker and loop started")
        except mqtt.MQTTException as e:
            print(f"MQTT connection error: {e}")

        # Create and start threads
        recording_active.set()
        recorder = threading.Thread(target=recording_thread)
        processor = threading.Thread(target=processing_thread)

        # Start threads
        recorder.start()
        processor.start()

        processor.join()

        # Keep the main thread running
        while True:
            pass

    except KeyboardInterrupt:
        print("\nStopping threads...")
        client.loop_stop()
        client.disconnect()
        recording_active.clear()
        recorder.join()
        processor.join()
        print("Threads stopped successfully")


if __name__ == "__main__":
    main()