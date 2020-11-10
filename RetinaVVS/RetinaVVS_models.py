from torch.utils.data import DataLoader, random_split
from pytorch_lightning import loggers as pl_loggers
import RetinaVVS.RetinaVVS_class as RetinaVVS_class
from torchvision.datasets import CIFAR10
from argparse import ArgumentParser
from torchvision import transforms
import pytorch_lightning as pl
import pandas as pd

if __name__ == "__main__":
    # Manual seeding
    pl.seed_everything(42)

    # Terminal Arguments
    parser = ArgumentParser()
    parser.add_argument('--no-auto_lr_find', dest='auto_lr_find', action='store_false')
    parser.set_defaults(auto_lr_find=True)
    parser.add_argument('--fast_dev_run', dest='fast_dev_run', action='store_true')
    parser.set_defaults(fast_dev_run=False)
    parser.add_argument("--es_patience", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    print(args)

    # Model hyperparameters
    hparams = {
        'batch_size': 32,
        'ret_channels': 32,
        'vvs_layers': 4,
        'dropout': 0.05,
        'model_class': "RetinaVVS",
        'lr': 1e-3
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
    model = getattr(RetinaVVS_class, hparams["model_class"])(hparams)

    # Callbacks
    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=args.es_patience,
        mode="min"
    )
    tb_logger = pl_loggers.TensorBoardLogger(f"RetinaVVS/logs/", name=model.name)

    # Train the model
    trainer = pl.Trainer.from_argparse_args(args, early_stop_callback=early_stop,
                                            deterministic=True, logger=tb_logger,
                                            default_root_dir="Models/")
    trainer.fit(model, train_dataloader=train_data, val_dataloaders=val_data)
