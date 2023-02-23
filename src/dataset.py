import os

from datasets import Audio, Dataset, DatasetDict, Features
from sklearn.model_selection import train_test_split
from transformers import AutoFeatureExtractor

from src.model import (
    Wav2Vec2ForSequenceMultiClassification,
    WhisperForSequenceClassification,
)
from src.utils import get_ds_name, unwrap_dataset, wrap_dataset

DATASET_FEATURES = [
    "genre",
    "category",
    "subcategory",
    "key type",
    "key signature",
    "time signature",
    "beat count",
    "Single",
    "Ensemble",
    "Dry",
    "Processed",
    "Clean",
    "Distorted",
    "Grooving",
    "Arrhythmic",
    "Acoustic",
    "Electric",
    "Melodic",
    "Dissonant",
    "Relaxed",
    "Intense",
    "Part",
    "Fill",
    "Cheerful",
    "Dark",
]

FEATURE_ENCODER_DETAILS = {
    "wav2vec2": {
        "class": Wav2Vec2ForSequenceMultiClassification,
        "pretrained": "facebook/wav2vec2-base",
    },
    "whisper": {
        "class": WhisperForSequenceClassification,
        "pretrained": "openai/whisper-tiny",
    },
}


_feature_extractors = {}


def get_feature_extractor(training_config):
    global _feature_extractors

    f_encoder = training_config["feature_encoder"]

    if f_encoder not in _feature_extractors:
        details = FEATURE_ENCODER_DETAILS[f_encoder]
        _feature_extractors[f_encoder] = AutoFeatureExtractor.from_pretrained(
            details["pretrained"]
        )

    return _feature_extractors[f_encoder]


def filter_df(df, features_config, remove_nones=True):
    print(f"{len(df)} total samples in dataset")

    drop_features = []
    if len(features_config) > 0:
        drop_features = [
            f
            for f in DATASET_FEATURES
            if f not in features_config.keys()
            if f in df.columns
        ]
        print(f"{len(DATASET_FEATURES) - len(drop_features)} features considered")

    # Using .mp3 files paths
    df["path"] = df["path"].apply(lambda x: x.replace(".caf", ".mp3"))
    # Using the filepath as ID for each sample
    df["id"] = df["path"].apply(
        lambda x: "_".join(".".join(x.split(".")[:-1]).replace("/", " ").split(" "))
    )
    df = df.drop(columns=drop_features)

    if remove_nones:
        df_nas = df.isna().any(axis=1)
        df = df[~df_nas]
        print(
            f"Considering only rows without non-available values, {df_nas.sum()} samples discarded"
        )

    for f_name, f_conf in features_config.items():
        top_n = f_conf["top_n"] if "top_n" in f_conf else None
        if top_n:
            df = df[df[f_name].isin(df.value_counts(f_name, sort=True)[:top_n].index)]
            print(f"Keeping only the {top_n} most frequent values of {f_name}")

    for f_name, f_conf in features_config.items():
        samples = f_conf["samples"] if "samples" in f_conf else None
        if samples:
            df = sample_df(df, f_name, total_samples=samples)
            print("Applying stratified sampling to the database")

    print(f"{len(df)} total samples left")
    return df


def sample_df(df, target_feature, total_samples):
    samples_per_value = total_samples // len(df[target_feature].unique())
    df_sampled = df.groupby(target_feature, group_keys=False).apply(
        lambda x: x.sample(min(len(x), samples_per_value))
    )
    df_sampled = df_sampled.reset_index(drop=True)
    print(f"Sampled {len(df_sampled)} items")
    return df_sampled


def split_df(df, validation_size, test_size=None):
    dfs = {}

    if test_size is None:
        test_size = 0

    dfs["train"], temp_df = train_test_split(
        df,
        test_size=test_size + validation_size,
    )

    if test_size:
        dfs["valid"], dfs["test"] = train_test_split(
            temp_df,
            test_size=test_size / (test_size + validation_size),
        )
    else:
        dfs["valid"] = temp_df

    for split, df_split in dfs.items():
        df.loc[df["id"].isin(df_split["id"]), "split"] = split

    return df


def get_dataset(df):
    return Dataset.from_pandas(df.reset_index(drop=True))


def prepare_ds(
    ds_source: Dataset,
    df_splits,
    feature_configs,
    fixed_mapping=None,
    save=False,
    original_path=None,
):
    drop_features = [f for f in DATASET_FEATURES if f not in feature_configs.keys()]

    if len(feature_configs) > 1:
        raise NotImplementedError()

    print("Removing extra columns from dataset")
    ds_source = ds_source.remove_columns(
        (["audio"] if "audio" in ds_source.features else [])
        + (["path"] if "audio" in ds_source.features else [])
        + [f for f in drop_features if f in ds_source.features]
    )

    ds = DatasetDict()
    id_to_index = {id_: i for i, id_ in enumerate(ds_source["id"])}

    for split in df_splits["split"].unique():
        print(f"Extracting {split} split")

        # Filter dataset `ds` by IDS present in `df`
        indices_to_keep = [
            id_to_index[id_]
            for id_ in set(df_splits[df_splits["split"] == split]["id"])
            if id_ in id_to_index
        ]
        ds[split] = ds_source.select(indices_to_keep)

    if fixed_mapping:
        # TODO: Should force the way features are casted to ClassLabels
        raise NotImplementedError()

    print("Create `ClassLabels` for target classes")
    ds = _cast_features(ds, df_splits, target_features=feature_configs.keys())

    for f in feature_configs.keys():
        ds = ds.rename_column(f, "label")

    if save:
        ds_path = get_ds_name(feature_configs, original_path)
        ds.save_to_disk(ds_path)
    return ds


def add_audio_column(ds, audios_dir_path, sampling_rate=None, training_config=None):
    if training_config and not sampling_rate:
        sampling_rate = get_feature_extractor(training_config).sampling_rate

    ds = wrap_dataset(ds)
    for split, ds_split in ds.items():
        ds[split] = ds_split.add_column(
            "audio", [os.path.join(audios_dir_path, path) for path in ds_split["path"]]
        ).cast_column(
            "audio",
            Audio(sampling_rate=sampling_rate),
        )

    return unwrap_dataset(ds)


def _get_features_dict(df, target_features):
    return {
        f: {
            "id": None,
            "names": sorted(df[f].unique()),
            "_type": "ClassLabel",
        }
        for f in target_features
    }


def _cast_features(ds, df, target_features):
    features = Features.from_dict(_get_features_dict(df, target_features))
    for f, args in features.items():
        ds = ds.cast_column(f, args)
    return ds
