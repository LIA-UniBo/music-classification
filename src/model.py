from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Model,
)
from transformers.models.whisper.modeling_whisper import (
    WhisperConfig,
    WhisperEncoder,
    WhisperPreTrainedModel,
)

WHISPER_INPUT_SEGMENTS = 3000


class ClassifierMLPHead(nn.Module):
    def __init__(self, embedding_size, hidden_layers, dropout):
        super().__init__()

        hidden_layers = [embedding_size] + hidden_layers
        self.layers = nn.ModuleList(
            [
                nn.Linear(hidden_layers[idx], hidden_layers[idx + 1])
                for idx in range(len(hidden_layers) - 1)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, hidden_state):
        for layer in self.layers:
            hidden_state = layer(hidden_state)
            hidden_state = self.dropout(hidden_state)
            hidden_state = self.relu(hidden_state)
        return hidden_state


class WhisperForSequenceClassification(WhisperPreTrainedModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)

        self.encoder = WhisperEncoder(config)

        # Classifier head
        self.head = ClassifierMLPHead(
            embedding_size=config.d_model,
            hidden_layers=config.classifier_hidden_layers,
            dropout=config.classifier_dropout,
        )
        self.classifier = nn.Linear(
            config.classifier_hidden_layers[-1], config.num_labels
        )

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_encoder(self):
        self.encoder._freeze_parameters()

    def forward(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Pad the input mel-spectrogram to have size BxCx3000
        # TODO: Move in dedicated DataCollator
        b, c, t = input_features.size()
        padded_input = torch.zeros(b, c, WHISPER_INPUT_SEGMENTS).to(
            input_features.device
        )
        padded_input[:, :, :t] = input_features
        input_features = padded_input

        outputs = self.encoder(
            input_features,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.head(hidden_states)
        pooled_output = hidden_states.mean(dim=1)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Wav2Vec2ForSequenceMultiClassification(Wav2Vec2ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)
        self.head = ClassifierMLPHead(
            embedding_size=config.hidden_size,
            hidden_layers=config.classifier_hidden_layers,
            dropout=config.classifier_dropout,
        )
        self.classifier = nn.Linear(
            config.classifier_hidden_layers[-1], config.num_labels
        )

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_encoder(self):
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.head(hidden_states)
        pooled_output = hidden_states.mean(dim=1)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
