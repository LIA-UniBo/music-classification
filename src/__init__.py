from transformers import AutoModelForSequenceClassification
from transformers.models.whisper import WhisperConfig

from src.model import WhisperForSequenceClassification

AutoModelForSequenceClassification.register(
    WhisperConfig, WhisperForSequenceClassification
)
