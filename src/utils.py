import time

import datasets
import IPython.display as ipd
import numpy as np
from scipy.io.wavfile import write


def convert_if_label(sample, feature, label_maps):
    value = sample[feature]
    if feature in label_maps:
        if type(value) == int:
            return label_maps[feature]["i2l"][value]
        else:
            return label_maps[feature]["l2i"][value]
    else:
        return value


def play_audio(audio):
    ipd.display(
        ipd.Audio(
            data=audio["array"],
            autoplay=False,
            rate=audio["sampling_rate"],
        )
    )


def play_audios(samples, label_maps=None, print_features=[]):
    for sample in samples:
        descr = []
        for f in print_features:
            descr.append(f"{f}: {convert_if_label(sample, f, label_maps)}")
        print(" - ".join(descr))
        play_audio(samples["audio"])


def play_random_audios(ds, quantity, print_features=[]):
    play_audios(
        ds.select(np.random.choice(range(len(ds)), quantity, replace=False)),
        # ds.select(np.random.randint(0, len(ds), quantity)),
        get_dataset_label_mapping(ds),
        print_features=print_features,
    )


def play_audios_by_id(ds, ids, print_features=[]):
    id_to_index = {id_: i for i, id_ in enumerate(ds["id"])}
    play_audios(
        ds.select([id_to_index[idx] for idx in ids]),
        get_dataset_label_mapping(ds),
        print_features=print_features,
    )


def wrap_dataset(ds):
    if type(ds) != "dict":
        ds = {"_": ds}
    return ds


def unwrap_dataset(ds):
    if len(ds) == 1 and "_" in ds.keys():
        ds = ds["_"]
    return ds


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


def get_csv_name(config, csv_path):
    path_pieces = [csv_path.split(".")[0]] + _get_file_suffixes(config)
    return "_".join(path_pieces) + ".csv"


def get_ds_name(config, ds_path):
    path_pieces = [ds_path] + _get_file_suffixes(config)
    return "_".join(path_pieces)


def get_model_name(config):
    freeze = "frz" if config["freeze_encoder"] else "fnt"
    layers = "_".join([str(layer) for layer in config["classifier_layers"]])
    dropout = str(config["classifier_dropout"]).split(".")[-1]
    return f"{config['feature_encoder']}-{freeze}-c{layers}-d{dropout}"


def get_run_name(config):
    return f"{get_model_name(config)}-{time.strftime('%Y%m%d-%H%M%S')}"


def _get_file_suffixes(config):
    suffix = []
    for feature_name, feature_config in config.items():
        substr = (
            feature_name
            + (f"{feature_config['top_n']}" if feature_config["top_n"] else "")
            + (f"s{feature_config['samples']}" if feature_config["samples"] else "")
        )
        suffix.append(substr)
    return suffix


def debug_sample_to_wav(audio: datasets.Audio, filename: str):
    data = np.array(audio["array"]).astype(np.int16)
    write(filename, audio["sampling_rate"], data)
