import warnings
from argparse import ArgumentParser
from os import path

import torch
import yaml
from lightning import Fabric
from torch.utils.data import DataLoader
from tqdm import tqdm

from dbnet.model_dbnet import DBNet
from dbnet.training import train
from dbnet.configs import Config
from dbnet.datasets import DBNetDataset
from dbnet.transform_albumen import get_augment

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, LambdaLR

# model = model_dbnet.DBNet("mobilenet_v3_large", 256, 1)
# train_data = "lmdb/train_data/"
# val_data = "lmdb/val_lmdb/"
# batch_size = 3
# augment = True


# train(model, train_data, val_data, augment=augment)
def loop_loader(loader, total_steps: int):
    """Loop over dataloader for some steps

    Args:
        loader:
            The dataloader to loop over
        total_steps:
            Number of steps to loop

    Yields:
        step:
            The current step (starts from 1).
            This would makes it easier to implement actions every n steps without
            having to exclude the first step.
        batch:
            Whatever the dataloader yield.
    """
    step = 0
    while True:
        for batch in loader:
            step = step + 1
            yield step, batch
            if step == total_steps:
                return


def DBNetScheduler(optimizer, total_steps: int, power: float = 0.9):
    """DBNet LR Scheduler"""

    def lr_dbnet(step):
        return (1 - step / total_steps) ** power

    return LambdaLR(optimizer, lr_dbnet)


def main_train(config: Config):
    # Context
    options = config.resolve()

    total_steps = config.total_steps
    print_every = config.print_every
    validate_every = config.validate_every

    lr = config.learning_rate
    train_data = config.train_data
    val_data = config.val_data
    batch_size = config.batch_size
    num_workers = config.num_workers
    weight_path = config.weight_path
    latest_weight_path = config.latest_weight_path

    # Model and optimization
    model = DBNet(**options["model"])
    if path.isfile(weight_path or "/idk"):
        model.load_state_dict(torch.load(weight_path, map_location="cpu"), strict=False)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    lr_scheduler = DBNetScheduler(optimizer, total_steps)

    # Load data
    augment = get_augment(**options["augment"])
    train_data = DBNetDataset(train_data, transform=augment)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    val_data = DBNetDataset(val_data)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers)

    # Setup fabric
    fabric = Fabric()
    model, optimizer = fabric.setup(model, optimizer)
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    # training loop
    pbar = tqdm(range(total_steps), "Training")
    model.train()
    for step, batch in loop_loader(train_loader, total_steps):
        optimizer.zero_grad()
        images, gt_probas, gt_thresholds = batch
        proba_maps, threshold_maps, loss = model(images, gt_probas, gt_thresholds)

        # Backward pass
        fabric.backward(loss)
        optimizer.step()
        lr_scheduler.step()

        # Logging
        loss_ = loss.item()
        lr_ = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({"loss": loss_, "lr": lr_})
        pbar.update()

        # Check pointing
        if step % print_every == 0:
            torch.save(model.state_dict(), latest_weight_path)
            tqdm.write(f"Model saved to {latest_weight_path}")

        if step % validate_every == 0:
            warnings.warn("Validation not implemented", Warning)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = Config(**config)
    main_train(config)
