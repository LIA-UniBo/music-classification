import math
import os
import time

import evaluate
import numpy as np
import wandb
from transformers import AutoConfig, DataCollatorWithPadding, Trainer, TrainingArguments
from transformers.trainer_utils import speed_metrics

from src.dataset import FEATURE_ENCODER_DETAILS, get_feature_extractor
from src.utils import get_feature_label_mapping

PROJECT_NAME = "music-classification-aii"
DEFAULT_MAX_AUDIO_LEN_MS = 10_000


def get_preprocess_func(training_config, max_audio_len_ms=DEFAULT_MAX_AUDIO_LEN_MS):
    feature_extractor = get_feature_extractor(training_config)

    def preprocess_function(examples):
        audio_arrays = [example["array"] for example in examples["audio"]]
        audio_sampling_rates = [
            example["sampling_rate"] for example in examples["audio"]
        ]
        sampling_rate = (
            audio_sampling_rates[0] if len(set(audio_sampling_rates)) == 1 else None
        )
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=sampling_rate,
            max_length=int(sampling_rate * max_audio_len_ms / 1000),
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
    fe_details = FEATURE_ENCODER_DETAILS[training_config["feature_encoder"]]

    config = AutoConfig.from_pretrained(fe_details["pretrained"])

    config.classifier_hidden_layers = training_config["classifier_layers"]
    config.classifier_dropout = training_config["classifier_dropout"]
    config.num_labels = class_feature.num_classes
    config.label2id = l2i
    config.id2label = i2l

    if "model_path" in training_config:
        model = fe_details["class"].from_pretrained(
            training_config["model_path"],
            config=config,
        )
    else:
        model = fe_details["class"].from_pretrained(
            fe_details["pretrained"],
            config=config,
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
    report_to="all",
):
    epochs = training_config["epochs"]
    train_batch_size = training_config["train_batch_size"]
    eval_batch_size = training_config["eval_batch_size"]
    learning_rate = training_config["learning_rate"]
    warmup_ratio = training_config["warmup"]

    if feature_extractor is None:
        feature_extractor = get_feature_extractor(training_config=training_config)

    if report_to:
        wandb.init(
            project=PROJECT_NAME,
            name=run_name,
            tags=([env] if env else []) + (["debug"] if debug else []),
        )

    data_collator = DataCollatorWithPadding(tokenizer=feature_extractor)

    training_args = TrainingArguments(
        run_name=run_name,
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=3,
        report_to=report_to,
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


def get_evaluator(
    model,
    training_config,
    feature_extractor=None,
    output_dir="out",
):
    epochs = training_config["epochs"]
    train_batch_size = training_config["train_batch_size"]
    eval_batch_size = training_config["eval_batch_size"]

    if feature_extractor is None:
        feature_extractor = get_feature_extractor(training_config=training_config)

    data_collator = DataCollatorWithPadding(tokenizer=feature_extractor)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
    )

    trainer = EvaluatorTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=feature_extractor,
        data_collator=data_collator,
        compute_metrics=get_metrics_func(),
    )

    return trainer


def end_training(run_name, trainer, models_dir_path):
    wandb.finish()
    trainer.save_model(os.path.join(models_dir_path, run_name))


class EvaluatorTrainer(Trainer):
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output
