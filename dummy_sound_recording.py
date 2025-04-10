import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from transformers import pipeline
import datetime
import librosa
import librosa.display
import matplotlib.pyplot as plt

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
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
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
    
def calculate_ptp(audio_data):
    """ Calculate the peak-to-peak value of the audio data """
    return np.ptp(audio_data)

def create_spectrogram(audio_data, sample_rate):
    """ Create and display a spectrogram from audio data """
    # Compute the Short-Time Fourier Transform (STFT)
    stft = librosa.stft(audio_data)
    # Convert the complex-valued STFT to magnitude
    spectrogram_data = np.abs(stft)

    # Display the spectrogram
    #plt.figure(figsize=(10, 6))
    #librosa.display.specshow(librosa.amplitude_to_db(spectrogram_data, ref=np.max),
    #                         sr=sample_rate, x_axis='time', y_axis='log')
    #plt.colorbar(format='%+2.0f dB')
    #plt.title('Spectrogram')
    #plt.show()

# Example usage:
set_start()
sample_rate, audio_data = record_audio(duration=5, save_to_file=SAVE_RECORDING)
labels_list = generate_labels_list()
result = audio_classification(audio_data, labels_list)
print(result)

# Calculate and print the ptp value
ptp_value = calculate_ptp(audio_data)
print(f"Peak-to-Peak value of the audio data: {ptp_value}")

# Create and display the spectrogram
create_spectrogram(audio_data, sample_rate)

