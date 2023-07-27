import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from lightning import Fabric
from tqdm import tqdm


from .datasets import DBNetDataset
from .transform_albumen import get_augment


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


def train(
    model: nn.Module,
    train_data: str,
    val_data: str,
    total_steps: int = 100_000,
    print_every: int = 250,
    batch_size: int = 1,
    num_workers: int = 1,
    augment: bool = False,
):
    """Train the model"""
    # Training helpers
    fabric = Fabric()

    # Load data
    transform = get_augment() if augment else None
    train_loader = DataLoader(
        DBNetDataset(train_data, transform=transform),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    val_loader = DataLoader(DBNetDataset(val_data), batch_size=batch_size)

    # Optimization
    optimizer = AdamW(model.parameters(), lr=7e-3)
    lr_scheduler = DBNetScheduler(optimizer, total_steps)

    # Setup
    model, optimizer = fabric.setup(model, optimizer)
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    # Training
    pbar = tqdm(range(total_steps), "Training")
    for step, batch in loop_loader(train_loader, total_steps):
        # Forward pass
        optimizer.zero_grad()
        images, gt_probas, gt_thresholds = batch
        proba_maps, threshold_maps, loss = model(images, gt_probas, gt_thresholds)

        # Backward pass
        fabric.backward(loss)
        optimizer.step()
        lr_scheduler.step()

        # Logging
        loss_item = loss.item()
        lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({"loss": loss_item, "lr": lr})
        pbar.update()

        if step % print_every == 0:
            torch.save(model.state_dict(), "latest.pt")
            tqdm.write("Model saved to latest.pt")
