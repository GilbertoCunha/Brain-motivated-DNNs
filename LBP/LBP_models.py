from torch.utils.data import DataLoader, random_split
from pytorch_lightning import loggers as pl_loggers
from torchvision.datasets import CIFAR10
from argparse import ArgumentParser
from torchvision import transforms
import pytorch_lightning as pl
import LBP.LBP_classes as LBP_classes
from functools import reduce
import pandas as pd
import optuna
import torch

if __name__ == "__main__":
    # Manual seeding
    pl.seed_everything(42)

    # Terminal Arguments
    parser = ArgumentParser()
    parser.add_argument("--model_class", type=str, default="LBPVVSEnd")
    parser.add_argument("--study_name", type=str, default="test")
    parser.add_argument("--es_patience", type=int, default=3)
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    # Optuna Hyperparameter Study
    hparams = {
        'model_class': [parser_args.model_class],
        'batch_size': 32,
        'ret_channels': 32,
        'vvs_layers': 4,
        'dropout': 0.05,
        'out_channels': 8,
        'kernel_size': 9,
        'sparsity': 0.8
    }

    # Train and validation dataloaders
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor()
    ])
    train_data = CIFAR10(root="data/CIFAR10", download=False, train=True, transform=transform)
    input_shape = tuple(train_data[0][0].shape)
    val_size = int(0.2 * len(train_data))
    train_size = len(train_data) - val_size
    train_data, val_data = random_split(train_data, [train_size, val_size])
    train_data = DataLoader(train_data, batch_size=hparams["batch_size"], shuffle=True, num_workers=12)
    val_data = DataLoader(val_data, batch_size=hparams["batch_size"], num_workers=12)

    # Create model
    hparams["input_shape"] = input_shape
    model = getattr(LBP_classes, hparams["model_class"])(hparams)

    # Callbacks
    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=args.es_patience,
        mode="min"
    )
    tb_logger = pl_loggers.TensorBoardLogger(f"LBP/logs/{model.filename}", name=model.name)

    # Train the model
    trainer = pl.Trainer.from_argparse_args(args, deterministic=True, early_stop_callback=early_stop,
                                            auto_lr_find=True, logger=tb_logger, fast_dev_run=False)
    trainer.fit(model, train_dataloader=train_data, val_dataloaders=val_data)