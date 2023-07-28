from argparse import ArgumentParser
from os import path

import torch
import yaml
from torch import nn

from dbnet.model_dbnet import DBNet
from dbnet.configs import Config


class WithSigmoid(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x, _, _ = self.model(x)
        return torch.sigmoid(x * 50)


def export(config: Config):
    # Context
    options = config.resolve()
    weight_path = config.weight_path or "/idk"
    output_file = config.onnx_path
    image_size = config.image_size
    assert path.exists(weight_path)
    assert output_file is not None

    # Load model
    model = DBNet(**options["model"])
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model = WithSigmoid(model)
    model = model.eval()

    # Example inputs
    inputs = torch.rand(1, 3, image_size[1], image_size[0])
    torch.onnx.export(
        model,
        inputs,
        output_file,
        input_names=["images"],
        dynamic_axes=dict(images=[0]),
        do_constant_folding=True,
    )
    print(f"Model exported to {output_file}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = Config(**config)
    export(config)
