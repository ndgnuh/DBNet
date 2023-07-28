from dataclasses import dataclass, field, fields, Field, _MISSING_TYPE
from typing import Tuple, Optional, Dict, Union, List
from inspect import cleandoc


import yaml


def config_field(help_string, choices: Optional[List] = None, **kwargs):
    """
    Create a dataclass field with doc string and choices as metadata.

    Args:
        help_string (str):
            The help string providing a description of the field's purpose and usage.
        choices (Optional[List], optional):
            A list of valid choices for the field. Default is None.
        **kwargs:
            Additional keyword arguments to pass to the field creation.

    Returns:
        Field:
            The dataclass field with the provided metadata.
    """
    help_string = cleandoc(help_string)
    metadata = dict(help_string=help_string, choices=choices)
    return field(metadata=metadata, **kwargs)


def get_config_field_str(fld: Field, default_value=None):
    """
    Generate a configuration string for a dataclass field.

    Args:
        fld (Field):
            The dataclass field for which the configuration string is generated.
            Create the field with `config_field` for `help_string` and `choices` metadata.
        default_value (Any, optional):
            The default value for the field. If not provided, the function will try to retrieve the default value from the field.

    Returns:
        str:
            The configuration string representing the dataclass field.
    """
    # Field name
    field_name = fld.name
    field_type = fld.type
    if isinstance(field_type, type):
        field_type = field_type.__name__

    # Default value
    if default_value is None:
        if callable(fld.default_factory):
            default_value = fld.default_factory()
        default_value = fld.default
    if isinstance(default_value, _MISSING_TYPE):
        default_value = None

    # Help strings
    possible_values = fld.metadata.get("choices", None)
    help_string = fld.metadata.get("help_string", None)

    # Lines
    lines = []

    # Field name
    lines.append(f"# {field_name} ({field_type}):")

    # Help string
    if help_string is not None:
        help_string = help_string.splitlines()
        lines.extend([f"#\t{line}" for line in help_string])

    # Choices
    if possible_values is not None:
        lines.append(f"#\tSupported values: {', '.join(map(str, possible_values))}")

    # Padding

    # Default value
    lines.append(yaml.dump({field_name: default_value}))
    config_str = "\n".join(lines)

    return config_str


@dataclass
class Config:
    # Model config
    image_size: Tuple[int, int] = config_field(
        """
        The size of the input image.
        The value must be a tuple of (width, height).
        """,
    )

    hidden_size: int = config_field(
        """
        The number of output channels for the feature pyramid module.
        The number of channels of prediction head will be 1/4 of this.
        """
    )

    classes: List[str] = config_field(
        """
        The object classes.
        Mapping from class index to string is required for pretty printting
        and for reverse mapping from dataset files.
        The model only cares about how many class there are.
        """
    )

    backbone: str = config_field(
        """
        Model backbone
        """,
        choices=["mobilenet_v3_large", "mobilenet_v3_small"],
        default="mobilenet_v3_large",
    )

    target_size: Optional[Tuple[int, int]] = config_field(
        """
        The size of the target heatmap.
        Can either be a tuple of (width, height) or null.
        If null, `image_size` will be used.
        """,
        default=None,
    )

    max_distance: Optional[Union[int, float]] = config_field(
        """
        The maximum shrinking distance for the bounding boxes.
        None value means no maximum.
        Integer value means absolute maximum value.
        Float value means the maximum will be some percentage of the target area.
        """,
        default=None,
    )

    min_box_size: Optional[Union[int, float]] = config_field(
        """
        The minimum bounding box size.
        None value means no minimum size.
        Integer value means absolute minimum size.
        Float value means the minimum size will be some percentage of the target area.
        """,
        default=None,
    )

    shrink: bool = config_field(
        """
        If `True`, the polygons are shrunken when drawing the probablity maps.
        If `False`, the original polygons are used.
        """,
        default=True,
    )

    shrink_rate: float = config_field(
        """
        DBNet shrink ratio from DBNet paper. The shrink will be
        A * (1 - r^2) / L, where r is the ratio, A and L are the
        area and the length of the polygon bounding box.
        """,
        default=0.4,
    )

    expand_rate: float = config_field(
        """
        DBNet expand ratio from DBNet paper. The expand distance will be
        A * r / L, where r is the ratio, A and L are the area
        and the length of the polygon bounding box.
        """,
        default=1.5,
    )

    # Training config
    train_data: Optional[str] = config_field(
        """
        Path to training data lmdb directory.
        Not used in inference process.
        Must not be null if training.
        """,
        default=None,
    )

    src_train_data: Optional[str] = config_field(
        """
        Path to train data index.
        This is used to create LMDB dataset.
        """,
        default=None,
    )

    val_data: Optional[str] = config_field(
        """
        Path to validation data lmdb directory.
        Not used in inference process.
        Must not be null if training.
        """,
        default=None,
    )

    src_val_data: Optional[str] = config_field(
        """
        Path to validate data index.
        This is used to create LMDB dataset.
        """,
        default=None,
    )

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
    config = Config(
        image_size=[1024, 1024],
        hidden_size=256,
        classes=["text"],
    )
    config_str = [
        get_config_field_str(field, getattr(config, field.name))
        for field in fields(config)
    ]
    return "\n".join(config_str)


def resolve_config(config: Config) -> Dict:
    # Encoding
    encode_options = {}
    encode_options["shrink"] = config.shrink
    encode_options["shrink_rate"] = config.shrink_rate
    encode_options["target_size"] = config.target_size or config.image_size
    encode_options["num_classes"] = config.num_classes
    encode_options["max_distance"] = config.max_distance
    encode_options["min_box_size"] = config.min_box_size

    # Model
    model_options = {}
    model_options["backbone"] = config.backbone
    model_options["hidden_size"] = config.hidden_size
    model_options["num_classes"] = config.num_classes

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
    options["model"] = model_options
    return options
