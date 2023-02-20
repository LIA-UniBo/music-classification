import os

import evaluate
import numpy as np
import wandb
from transformers import (
    AutoModelForAudioClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.dataset import (
    FEATURE_ENCODER_TO_HF_HUB,
    get_feature_extractor,
    get_feature_label_mapping,
)

PROJECT_NAME = "music-classification-aii"
MAX_AUDIO_LEN_S = (
    10  # TODO: Transformers are O(n^2) so high audio len could be prohibitive
)


def get_preprocess_func(training_config):
    feature_extractor = get_feature_extractor(training_config)

    def preprocess_function(examples):
        audio_arrays = [example["array"] for example in examples["audio"]]
        audio_sampling_rates = [
            example["sampling_rate"] for example in examples["audio"]
        ]
        sampling_rate = (
            audio_sampling_rates[0] if len(set(audio_sampling_rates)) == 1 else None
        )
        # TODO: Add dynamic padding with DataCollator
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=sampling_rate,
            max_length=sampling_rate * MAX_AUDIO_LEN_S * 1000,
            truncation=True,
        )
        return inputs

    return preprocess_function


def get_metrics_func():
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

    return compute_metrics


def get_model(training_config, ds):
    class_feature = ds.features["label"]
    l2i, i2l = get_feature_label_mapping(class_feature)

    model = AutoModelForAudioClassification.from_pretrained(
        FEATURE_ENCODER_TO_HF_HUB[training_config["feature_encoder"]],
        num_labels=class_feature.num_classes,
        label2id=l2i,
        id2label=i2l,
    )

    if training_config["freeze_encoder"]:
        model.freeze_feature_encoder()

    return model


def get_trainer(
    run_name,
    model,
    train_ds,
    eval_ds,
    training_config,
    feature_extractor=None,
    output_dir="out",
    debug=False,
    env=None,
):
    epochs = training_config["epochs"]
    train_batch_size = training_config["train_batch_size"]
    eval_batch_size = training_config["eval_batch_size"]

    wandb.init(
        project=PROJECT_NAME,
        name=run_name,
        tags=([env] if env else []) + (["debug"] if debug else []),
    )

    data_collator = DataCollatorWithPadding()

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
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=feature_extractor,
        data_collator=data_collator,
        compute_metrics=get_metrics_func(),
    )

    return trainer


def end_training(run_name, trainer, models_dir_path):
    wandb.finish()
    trainer.save_model(os.path.join(models_dir_path, run_name))
