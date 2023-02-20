import os

import datasets
from datasets import Audio, Dataset, Features
from sklearn.model_selection import train_test_split
from transformers import AutoFeatureExtractor

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

FEATURE_ENCODER_TO_HF_HUB = {"wav2vec2": "facebook/wav2vec2-base"}


_feature_extractor = None


def get_feature_extractor(training_config):
    global _feature_extractor

    if not _feature_extractor:
        _feature_extractor = AutoFeatureExtractor.from_pretrained(
            FEATURE_ENCODER_TO_HF_HUB[training_config["feature_encoder"]]
        )

    return _feature_extractor


def filter_df(df, audios_dir_path, features_config, remove_nones=True):
    print(f"{len(df)} total samples in dataset")

    drop_features = []
    if len(features_config) > 0:
        drop_features = [f for f in DATASET_FEATURES if f not in features_config.keys()]
        print(f"{len(DATASET_FEATURES) - len(drop_features)} features considered")

    # Using .mp3 files paths, base directory is prepended
    df["path"] = df["path"].apply(lambda x: x.replace(".caf", ".mp3"))
    df["full_path"] = df["path"].apply(lambda x: os.path.join(audios_dir_path, x))
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

    return dfs


def get_dataset(df):
    return Dataset.from_pandas(df.drop(columns=["full_path"]).reset_index(drop=True))


def prepare_ds(
    ds: Dataset,
    df,
    feature_configs,
    test_split_size,
    fixed_mapping=None,
    save=False,
    original_path=None,
):
    drop_features = [f for f in DATASET_FEATURES if f not in feature_configs.keys()]

    # Filter dataset `ds` by IDS present in `df`
    id_to_index = {id_: i for i, id_ in enumerate(ds["id"])}
    indices_to_keep = [id_to_index[id_] for id_ in set(df["id"]) if id_ in id_to_index]
    ds = ds.select(indices_to_keep)
    ds = ds.remove_columns(["audio", "path", "full_path"] + drop_features)

    if fixed_mapping:
        raise NotImplementedError()
        # TODO: Should force the way features are casted to ClassLabels
    ds = _cast_features(ds, df, target_features=feature_configs.keys())

    if len(feature_configs) > 1:
        # TODO
        raise NotImplementedError()

    ds = ds.train_test_split(test_size=test_split_size)

    # TODO ds = ds.rename_column(target_features[0], "label")
    if save:
        ds_path = get_ds_name(feature_configs, original_path)
        ds.save_to_disk(ds_path)
    return ds


def add_audio_column(ds, training_config, audios_dir_path):
    ds = wrap_dataset(ds)
    for split, ds_split in ds.items():
        ds[split] = ds_split.add_column(
            "audio", [audios_dir_path + path for path in ds_split["path"]]
        ).cast_column(
            "audio",
            Audio(sampling_rate=get_feature_extractor(training_config).sampling_rate),
        )

    return unwrap_dataset(ds)


def get_feature_label_mapping(feature):
    label2id = {name: feature.str2int(name) for name in feature.names}
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label


def get_dataset_label_mapping(ds):
    label_maps = {}
    for f in ds.features:
        if type(ds.features[f]) == datasets.features.ClassLabel:
            m = get_feature_label_mapping(ds.features[f])
            label_maps[f] = {"l2i": m[0], "i2l": m[1]}
    return label_maps


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
