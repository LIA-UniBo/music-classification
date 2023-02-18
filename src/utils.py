import IPython.display as ipd
import numpy as np

from src.dataset import get_label_mappings


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


def play_random_audios(ds, quantity, print_features=[]):
    play_audios(
        ds.select(np.random.randint(0, len(ds), quantity)),
        get_label_mappings(ds),
        print_features=print_features,
    )
