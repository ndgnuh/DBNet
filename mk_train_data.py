from os import path, makedirs
from argparse import ArgumentParser
from typing import List, Tuple, Dict

import yaml
from tqdm import tqdm

from dbnet.io_utils import dataset_to_lmdb
from dbnet.parse_utils import parse_labelme_list
from dbnet.transform_dbnet import encode_dbnet
from dbnet.configs import Config, resolve_config


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("config", help="Config file")
    return parser.parse_args()


def mk_train_data(
    src_path: str,
    lmdb_path: str,
    classes: List[str],
    dbnet_config: Dict,
):
    # Load data
    root = path.dirname(lmdb_path)
    num_classes = len(classes)
    makedirs(root, exist_ok=True)
    dataset = parse_labelme_list(src_path, classes)

    # Encode the dataset
    def encoded_dataset():
        for sample in tqdm(dataset, "Building dataset"):
            encoded = encode_dbnet(*sample, **dbnet_config)
            yield encoded

    dataset_to_lmdb(lmdb_path, encoded_dataset(), num_classes)


def main():
    args = parse_args()
    config_file = args.config

    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = Config(**config)

    # Encoding config
    enc_config = resolve_config(config)["encoding"]
    mk_train_data(
        config.src_train_data,
        config.train_data,
        config.classes,
        enc_config,
    )
    mk_train_data(
        config.src_val_data,
        config.val_data,
        config.classes,
        enc_config,
    )


if __name__ == "__main__":
    main()
