This repo contains files for testing and running CLAP, an AI model for audio. 

## Introduction
Sensemakers is an Amsterdam based network around new technologies, IoT and AI. We have since long (2019) a project running on 'Urban Sounds', to investigate noise pollution in the city of Amsterdam. We wanted to test CLAP and in this repo is the code for doing so. 

### CLAP
CLAP (Contrastive Language-Audio Pretraining) is a neural network trained on a variety of (audio, text) pairs. It can be instructed in to predict the most relevant text label given an audio sample.

In this repo we have tried two different models:
1. larger_clap_music_and_speech
2. larger_clap_general

Conclusion: larger_clap_general gave the best results for our use case.

Sources:
- [CLAP on Huggingface](https://huggingface.co/laion/larger_clap_general)
- [CLAP paper](https://arxiv.org/abs/2211.06687)


### The dataset
This dataset contains nine classes of audio events in an urban environment. Contains 216 samples. The dataset is located on the 🤗 Huggingface Hub: 

[UrbanSoundsNew](https://huggingface.co/datasets/UrbanSounds/UrbanSoundsNew) 


## The repo folders

### 1. Notebooks for testing CLAP

Two notebooks using the UrbanSoundsII dataset:

1. UrbanSoundsII dataset with CLAP.ipynb. Run this notebook to test the CLAP model on the UrbanSounds dataset.
2. CLAP embeddings.ipynb. Get the audio embeddings, do similarity search and visualise the dataset using PCA/t-SNE.

Two notebooks using the UrbanSoundsSample dataset:

3. Real UrbanSoundsSamples with CLAP.ipynb
4. Visualising the UrbanSoundsSamples with CLAP embeddings.ipynb (in progress)

UrbanSounds_audio_pipeline.ipynb

We run CLAP with the 🤗 ```transformers``` library. Please find more info: [Huggingface CLAP](https://huggingface.co/docs/transformers/model_doc/clap)

### 2. Python files for Raspberry
Use the latest version:  **urban_sounds_3.5.py**

We use a config.py file with the credentials for MQTT. (Obviously, we do not store this file on GitHub)

We use **sound_scapes.py** to store the labels. For a given location, we can create a set of labels that we can classify. 
Currently, there is one location: 'Marineterrein'.

There are several settings in the python script, check the script before using it!

Also, this folder contains a sub-folder called 'cpu_usage', which contains a vibe-coded script that measures cpu_usage and creates a matplotlib graph as .png. Some results are also in this sub-folder. 

### 3. white noise experiments
Some experiments with (vibe coded) jupyter notebooks to add white noise to a sample and see how it influences the results of the audio classification. TLDR: Even with 100% white noise added, results of audio classification is still good.

### 4. CLAP documentation
Provides images, a powerpoint and clips about CLAP. 

### CLAP Video
[Watch the video on YouTube](https://youtu.be/dPcVhHVIoIs)



