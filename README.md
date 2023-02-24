---
Title: Music Classification
Author: Alessandro Stockman
---

# Music Classification

Comparison between `Wav2Vec2` and `Whisper` fine-tuned for music genre classification

## Project Structure

- 1_exploration.ipynb   | Analysis of the dataset
- 2_training.ipynb      | Training of different architectures
- 3_evaluation.ipynb    | Evaluations on the trained models

## Requirements

Tested with `python 3.10.6` and the following libraries:
- pandas
- matplotlib
- torch
- datasets
- transformers
- evaluations
- torchaudio
- wandb
- librosa
- evaluate
- tqdm
- scikit-learn
- ffmpeg-python

System dependencies:
- ffmpeg
- sox