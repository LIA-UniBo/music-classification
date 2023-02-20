from transformers import AutoModelForAudioClassification
from transformers.models.whisper import WhisperConfig

from src.model import WhisperForSequenceClassification

AutoModelForAudioClassification.register(
    WhisperConfig, WhisperForSequenceClassification
)
