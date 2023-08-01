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

Open the config file and change `classes`, `train_data`, `src_train_data`, `src_val_data` and `val_data`,
See the `examples` directory for example data and config.

Then, create the training and validation dataset.
```shell
python mk_train_data.py config.yml
```

Optionally, fire up jupyter notebook and see the [Visualize.ipynb](https://github.com/ndgnuh/DBNet/blob/master/Visualize.ipynb) to view the data before training. These are the data that will be fed directly into the model.

Finally, train the model.
```shell
python train.py config.yml
```

### Logging

Logs will be written to `runs` directory. Run `tensorboard` to view the logs
```shell
tensorboard --log_dir runs --bind_all
```

### Export to ONNX


Change the path to output onnx in `config.yaml`, then run
```shell
python export-onnx.py config.yaml
```

Inference example:
```python
from dbnet import DBNetONNX
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import numpy as np

model = DBNetONNX("./db_mobilenet_v3_large.onnx")

image = Image.open("my-image.png")
image_draw = ImageDraw.Draw(image)
results = model.predict(image)
boxes = results['boxes']
classes = results['classes']
colors = [(255,0, 0), (0, 255, 0), (0, 0, 255)]
for polygon, class_idx in zip(boxes, classes):
    color = colors[class_idx]
    image_draw.polygon(polygon, outline=color)
plt.imshow(np.concatenate(results['proba_maps'], axis=1))
plt.show()
plt.imshow(image)
plt.show()
```

Installation as a dependency (for inference without torch): TODO.
