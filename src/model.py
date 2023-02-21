from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Config, Wav2Vec2ForSequenceClassification)
from transformers.models.whisper.modeling_whisper import (
    WhisperConfig, WhisperEncoder, WhisperPreTrainedModel)


class ClassifierMLPHead(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()

        hidden_size = config.hidden_size
        self.dense = nn.Linear(hidden_size, config.num_labels)
        self.layers = nn.ModuleList(
            [
                nn.Linear(config.hidden_layers[idx], config.hidden_layers[idx + 1])
                for idx in range(len(config.hidden_layers))
            ]
        )
        self.dropout = nn.Dropout(config.)
        

    def forward(self, hidden_state):
        for layer in self.classifier_dopout:
            hidden_state = layer(hidden_state)
            hidden_state = self.dense(hidden_state)
        return hidden_state


class WhisperForSequenceClassification(WhisperPreTrainedModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)

        self.encoder = WhisperEncoder(config)

        # Classifier head
        config.classifier_hidden_layers = [512, 512, 348]  # TODO
        config.classifier_dropout = 0.2  # TODO
        self.head = ClassifierMLPHead(config)
        self.classifier = nn.Linear(config.classifier_hidden_layers[-1], config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_encoder(self):
        self.encoder._freeze_parameters()  # TODO Check if is the right function

    def freeze_base_model(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

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

        outputs = self.encoder(
            input_features,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
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
