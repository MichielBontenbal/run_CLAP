In this repo you will find code to run the CLAP model with data from the Urban Sounds project. 

### CLAP
CLAP (Contrastive Language-Audio Pretraining) is a neural network trained on a variety of (audio, text) pairs. It can be instructed in to predict the most relevant text label given an audio sample.

In this repo we have tried two different models:
1. larger clap music and speech
2. larger clap general

The larger clap general model seems to give better results. 

Sources:
- [CLAP on Huggingface](https://huggingface.co/laion/larger_clap_general)
- [CLAP paper](https://arxiv.org/abs/2211.06687)


### The datasets
There are two dataset hosted on the Huggingface Hub.

1. [UrbanSoundsNew](https://huggingface.co/datasets/UrbanSounds/UrbanSoundsNew)
2. [UrbanSoundsSamples](https://huggingface.co/datasets/UrbanSounds/UrbanSoundsSamples)

**UrbanSoundsII**

A new version of an old dataset as the original dataset got corrupted. 
This dataset contains nine classes of audio events in an urban environment. 

**UrbanSoundsSamples**

This small dataset contains 50 samples of real audio events we collected at the location.
(It is private dataset as we need to preserve privacy).

We use 🤗  ```datasets``` library to load this dataset. 

### The notebooks

Two notebooks using the UrbanSoundsII dataset:

1. UrbanSoundsII dataset with CLAP.ipynb. Run this notebook to test the CLAP model on the UrbanSounds dataset.
2. CLAP embeddings.ipynb. Get the audio embeddings, do similarity search and visualise the dataset using PCA/t-SNE.

Two notebooks using the UrbanSoundsSample dataset:

3. Real UrbanSoundsSamples with CLAP.ipynb
4. Visualising the UrbanSoundsSamples with CLAP embeddings.ipynb (in progress)

5. A notebook to create an audio pipeline: UrbanSounds_audio_pipeline.ipynb

We run CLAP with the 🤗 ```transformers``` library. Please find more info: [Huggingface CLAP](https://huggingface.co/docs/transformers/model_doc/clap)

### The python script
Finally, the resulting python script that we will use on the new Raspberry Pi.

### CLAP Video
[Watch the video on YouTube](https://youtu.be/dPcVhHVIoIs)



