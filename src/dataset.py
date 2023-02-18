import os

import datasets
from datasets import Audio, Dataset, Features

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


def filter_df(df, keep_features, audios_dir_path, top_n=None, samples=None):
    drop_features = [f for f in DATASET_FEATURES if f not in keep_features]
    print(f"Only {len(DATASET_FEATURES) - len(drop_features)} features considered")

    print("Using .mp3 files paths")
    df["mp3_path"] = df["path"].apply(
        lambda x: os.path.join(audios_dir_path, x.replace(".caf", ".mp3"))
    )
    df["id"] = df["path"].apply(
        lambda x: "".join(x.split(".")[:-1]).replace("/", "_").replace(" ", "_")
    )
    df = df.drop(columns=["path"] + drop_features)

    df_nans = df.isna().any(axis=1)
    df = df[~df_nans]
    print(f"Considering only rows without NaNs, {df_nans.sum()} samples discarded")
    print(f"{len(df)} samples left")

    if top_n:
        for f, n in top_n.items():
            # TODO: What happens with multiple top_n?
            df = df[df[f].isin(df.value_counts(f, sort=True)[:n].index)]
            print(f"Keeping only the {n} most frequent values of {f}")
            print(f"{len(df)} samples left")

    if samples:
        # TODO: What should happen with multiple keep_features?
        df = sample_df(df, keep_features[0], total_samples=samples)
        print("Applying stratified sampling to the database")
        print(f"{len(df)} samples left")

    return df


def get_features_dict(df, dataset_features):
    features = {
        "id": {"dtype": "string", "id": None, "_type": "Value"},
        "mp3_path": {"dtype": "string", "id": None, "_type": "Value"},
    }

    for f in dataset_features:
        if f in df:
            features.update(
                {
                    f: {
                        "id": None,
                        "names": df[f].unique().tolist(),
                        "_type": "ClassLabel",
                    }
                }
            )

    return features


def get_dataset(df, dataset_features=DATASET_FEATURES):
    features = Features.from_dict(get_features_dict(df, dataset_features))
    ds = Dataset.from_pandas(
        df.reset_index(drop=True), features=features
    )  # TODO: Save as strings and convert to ClassLabels only in "prepare_df" - how to keep same ids?
    ds = ds.rename_column("mp3_path", "audio_path")
    # ds = ds.add_column(name="audio", column=df["mp3_path"])
    # ds = ds.cast_column("audio", Audio(
    #     # sampling_rate=16_000 #TODO: Depends on the model
    # ))

    return ds


def get_label_mappings(ds):
    label_maps = {}
    for f in ds.features:
        if type(ds.features[f]) == datasets.features.ClassLabel:
            m = create_label_maps(ds.features[f])
            label_maps[f] = {"l2i": m[0], "i2l": m[1]}
    return label_maps


def sample_df(df, target_feature, total_samples):
    samples_per_value = total_samples // len(df[target_feature].unique())
    df_sampled = df.groupby(target_feature, group_keys=False).apply(
        lambda x: x.sample(min(len(x), samples_per_value))
    )
    df_sampled = df_sampled.reset_index(drop=True)
    print(f"Sampled {len(df_sampled)} items")
    return df_sampled


def create_label_maps(feature):
    label2id = {name: feature.str2int(name) for name in feature.names}
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label


def get_preprocess_func(feature_extractor):
    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=16000,
            truncation=True,
        )
        return inputs

    return preprocess_function


def add_audio_column(ds):
    for split, ds_split in ds.items():
        ds[split] = ds_split.add_column("audio", ds_split["audio_path"]).cast_column(
            "audio",
            Audio(
                # sampling_rate=16_000 #TODO: Depends on the model
            ),
        )
    return ds
