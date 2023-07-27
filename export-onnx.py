import torch
from torch import nn
from dbnet import model_dbnet
from dbnet.training import train


class WithSigmoid(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x, _, _ = self.model(x)
        return torch.sigmoid(x * 50)


image_size = (1024, 1024)
hidden_size = 256
num_classes = 1
weight = "latest.pt"
backbone = "mobilenet_v3_large"
output_file = "db_mobilenet_v3_large.onnx"

model = model_dbnet.DBNet(backbone, hidden_size, num_classes)
model.load_state_dict(torch.load(weight, map_location="cpu"))
model = WithSigmoid(model)
model.eval()

inputs = torch.rand(1, 3, *image_size)
torch.onnx.export(
    model,
    inputs,
    output_file,
    input_names=["images"],
    dynamic_axes=dict(images=[0]),
    do_constant_folding=True,
)
