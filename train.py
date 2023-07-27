import torch
from dbnet import model_dbnet
from dbnet.training import train

model = model_dbnet.DBNet("mobilenet_v3_large", 256, 2)
train_data = "lmdb/train_data/"
val_data = "lmdb/val_lmdb/"
batch_size = 3


train(model, train_data, val_data)
