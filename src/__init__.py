from transformers import AutoModelForAudioClassification
from transformers.models.wav2vec2 import Wav2Vec2Config
from transformers.models.whisper import WhisperConfig

from src.model import (
    Wav2Vec2ForSequenceMultiClassification,
    WhisperForSequenceClassification,
)

AutoModelForAudioClassification.register(
    WhisperConfig, WhisperForSequenceClassification
)
