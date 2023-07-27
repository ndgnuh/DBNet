from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Union, List


@dataclass
class Config:
    # Model config
    image_size: Tuple[int, int]
    hidden_size: int
    classes: List[str]
    target_size: Optional[Tuple[int, int]] = None
    max_distance: Optional[Union[int, float]] = None
    min_box_size: Optional[Union[int, float]] = None
    shrink: bool = True
    shrink_rate: float = 0.4
    expand_rate: float = 1.5

    # Training config
    train_data: Optional[str] = None
    val_data: Optional[str] = None
    training_lr: float = 7e-3
    training_total_steps: int = 100_000
    training_batch_size: int = 1
    training_num_workers: int = 0
    training_print_every: int = 250
    training_augment: bool = True
    training_augment_rotation: bool = True
    training_augment_flip: bool = False

    @property
    def num_classes(self) -> int:
        return len(self.classes)


def example_config():
    return Config(
        image_size=(1024, 1024),
        hidden_size=256,
        classes=["text"],
    )


def resolve_config(config: Config) -> Dict:
    # Encoding
    encode_options = {}
    encode_options["shrink"] = config.shrink
    encode_options["shrink_rate"] = config.shrink_rate
    encode_options["target_size"] = config.target_size or config.image_size

    # Training
    train_options = {}
    train_options["train_data"] = config.train_data
    train_options["val_data"] = config.val_data
    train_options["lr"] = config.training_lr
    train_options["total_steps"] = config.training_total_steps
    train_options["batch_size"] = config.training_batch_size
    train_options["num_workers"] = config.training_num_workers
    train_options["print_every"] = config.training_print_every
    train_options["augment"] = config.training_augment
    train_options["augment_rotation"] = config.training_augment_rotation
    train_options["augment_flip"] = config.training_augment_flip

    # Option hub
    options = {}
    options["encoding"] = encode_options
    options["training"] = train_options
    return options
