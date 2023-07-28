# DBNet

### Getting started

Install requirements:
```shell
pip install -r requirements.txt
```

### Training

Generate config file:
```shell
python gen_config.py -o config.yaml
```

Open the config file and change `classes`, `train_data` and `val_data`.
See the `examples` directory for example data and config.

Then, create the training and validation dataset.
```shell
python mk_train_data.py -c examples/config.yaml \
    --train examples/train.txt \
    --val examples/val.txt
```

Finally, train the model (TODO).
