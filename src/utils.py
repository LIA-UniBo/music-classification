import IPython.display as ipd
import numpy as np


def convert_if_label(sample, feature, label_maps):
    value = sample[feature]
    if feature in label_maps:
        if type(value) == int:
            return label_maps[feature]["i2l"][value]
        else:
            return label_maps[feature]["l2i"][value]
    else:
        return value


def play_audios(samples, label_maps, print_features=[]):
    for sample in samples:
        descr = []
        for f in print_features:
            descr.append(f"{f}: {convert_if_label(sample, f, label_maps)}")
        print(" - ".join(descr))
        ipd.display(
            ipd.Audio(
                data=sample["audio"]["array"],
                autoplay=False,
                rate=sample["audio"]["sampling_rate"],
            )
        )


def play_random_audios(ds, label_maps, quantity, print_features=[]):
    play_audios(
        ds.select(np.random.randint(0, len(ds), quantity)),
        label_maps,
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


def get_csv_name(config, csv_path):
    path_pieces = [csv_path.split(".")[0]] + _get_file_suffixes(config)
    return "_".join(path_pieces) + ".csv"


def get_ds_name(config, ds_path):
    path_pieces = [ds_path] + _get_file_suffixes(config)
    return "_".join(path_pieces)


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
