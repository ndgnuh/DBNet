from argparse import ArgumentParser
from dbnet.configs import example_config

parser = ArgumentParser()
parser.add_argument("-o", dest="output")
args = parser.parse_args()
config_string = example_config()

if args.output is not None:
    with open(args.output, "w") as fp:
        fp.write(config_string)
    print(f"Config written to {args.output}")
else:
    print(config_string)
