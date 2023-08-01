from os import path
from argparse import ArgumentParser
import yaml
from dbnet.configs import example_config

parser = ArgumentParser()
parser.add_argument("-o", dest="output")
args = parser.parse_args()

if args.output is not None:
    name = path.splitext(args.output)[0]
    kwargs = {}
    kwargs["weight_path"] = kwargs["latest_weight_path"] = f"{name}.latest.pt"
    kwargs["best_weight_path"] = f"{name}.best.pt"
    kwargs["onnx_path"] = f"{name}.onnx"

    config_string = example_config(**kwargs)
    with open(args.output, "w") as fp:
        fp.write(config_string)
        print(f"Config written to {args.output}")
else:
    config_string = example_config()
    print(config_string)
