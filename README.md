In this repo you will find code to run the CLAP model with data from the Urban Sounds project. 

### CLAP
CLAP (Contrastive Language-Audio Pretraining) is a neural network trained on a variety of (audio, text) pairs. It can be instructed in to predict the most relevant text label given an audio sample.

Source: [!huggingface page](https://huggingface.co/laion/larger_clap_general)
CLAP paper: (https://arxiv.org/abs/2211.06687)

In this notebook we will use two CLAP models:
1. larger clap music and speech
2. larger clap general

In general I believe the larger clap general model gives better results. 

### The datasets
The dataset is hosted on the Huggingface Hub at: 
![UrbanSoundsII](https://huggingface.co/datasets/MichielBontenbal/UrbanSoundsII)

(This is a new version of the same dataset as the old dataset got corrupted.)

This dataset contains nine classes of audio events in an urban environment. 

In this notebook we will use 🤗  ```dataset``` library to load this dataset. 

And we'll use the 🤗 ```transformers``` library to run the CLAP model. Please find more info: https://huggingface.co/docs/transformers/model_doc/clap 
