import os

import evaluate
import numpy as np
import wandb
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Trainer,
    TrainingArguments,
)

from src.dataset import get_feature_label_mapping

PROJECT_NAME = "music-classification-aii"

FEATURE_ENCODER_TO_HF_HUB = {"wav2vec2": "facebook/wav2vec2-base"}


def get_metrics_func():
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

    return compute_metrics


def get_model_config_examples():
    c1 = {
        "feature_encoder": "wav2vec2",  # or "jukebox", "whisper", ...
        "freeze_encoder": True,
        "classifier": {"hidden_layers": [512], "dropout": 0.3, "heads": ["genre"]},
    }

    c2 = {
        "feature_encoder": "wav2vec2",  # or "jukebox", "whisper", ...
        "freeze_encoder": True,
        "classifier": {"hidden_layers": [512, 512], "dropout": 0.5, "heads": ["genre"]},
    }

    return c1, c2


def get_model(model_config, ds):
    class_feature = ds.features["label"]
    l2i, i2l = get_feature_label_mapping(class_feature)
    model = AutoModelForAudioClassification.from_pretrained(
        FEATURE_ENCODER_TO_HF_HUB[model_config["feature_encoder"]],
        num_labels=class_feature.num_classes,
        label2id=l2i,
        id2label=i2l,
    )

    if model_config["freeze_encoder"]:
        model.freeze_feature_encoder()

    return model


def get_trainer(
    run_name, model, ds, feature_extractor=None, output_dir="out", debug=False, env=None
):
    epochs = 10
    train_batch_size = 256
    eval_batch_size = 512

    wandb.init(
        project=PROJECT_NAME,
        name=run_name,
        tags=([env] if env else []) + (["debug"] if debug else []),
    )

    training_args = TrainingArguments(
        run_name=run_name,
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=3,
        report_to="all",
        logging_steps=50,
        logging_first_step=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=feature_extractor,
        compute_metrics=get_metrics_func(),
    )

    return trainer


def run_trainer(run_name, trainer, models_dir_path):
    trainer.train()
    wandb.finish()
    trainer.save_model(os.path.join(models_dir_path, run_name))
