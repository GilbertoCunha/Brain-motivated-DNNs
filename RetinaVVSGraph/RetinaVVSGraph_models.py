from torch.utils.data import DataLoader, random_split
from pytorch_lightning import loggers as pl_loggers
import RetinaVVSGraph.RetinaVVSGraph_class as RetinaVVSG
from argparse import ArgumentParser, Namespace
from torchvision.datasets import CIFAR10
from torchvision import transforms
import pytorch_lightning as pl
import pandas as pd
import optuna
import torch


def objective(trial, args, search):
    # Optuna trial parameters
    params = {key: trial.suggest_categorical(key, value) for key, value in search.items() if key != "vvs_graph"}
    params["vvs_graph"] = search["vvs_graph"]
    params["lr"] = 1e-3

    # Train and validation dataloaders
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor()
    ])
    train_data = CIFAR10(root="data/CIFAR10", download=False, train=True, transform=transform)
    input_shape = tuple(train_data[0][0].shape)
    val_size = int(0.2 * len(train_data))
    train_size = len(train_data) - val_size
    train_data, val_data = random_split(train_data, [train_size, val_size],
                                        generator=torch.Generator().manual_seed(trial.number))
    train_data = DataLoader(train_data, batch_size=params["batch_size"], shuffle=True, num_workers=12)
    val_data = DataLoader(val_data, batch_size=params["batch_size"], num_workers=12)

    # Create model
    params["input_shape"] = input_shape
    model = getattr(RetinaVVSG, params["model_class"])(params)

    # Callbacks
    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=args.es_patience,
        mode="min"
    )
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        filepath=f"RetinaVVSGraph/models",
        prefix=model.name,
        monitor="val_acc",
        mode="max",
        save_top_k=1
    )
    tb_logger = pl_loggers.TensorBoardLogger(f"RetinaVVSGraph/logs/", name=model.name)

    # Train the model
    trainer = pl.Trainer.from_argparse_args(args, early_stop_callback=early_stop, num_sanity_val_steps=0,
                                            checkpoint_callback=model_checkpoint, auto_lr_find=True,
                                            logger=tb_logger, fast_dev_run=False, max_epochs=100)
    trainer.fit(model, train_dataloader=train_data, val_dataloaders=val_data)

    # Save model state dict
    checkpoint = torch.load(model_checkpoint.best_model_path)
    model.load_state_dict(checkpoint["state_dict"])
    torch.save(model.state_dict(), f"RetinaVVSGraph/state_dicts/{model.name}.tar")

    return model_checkpoint.best_model_score


if __name__ == "__main__":
    # Terminal Arguments
    parser = ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--es_patience", type=int, default=3)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--study_name", type=str, default="test")
    parser_args = parser.parse_args()

    # VVS Network graph
    vvs_graph = {
    '0': [1],
    '1': [2, 4],
    '2': [3, 5],
    '3': [4, 6],
    '4': [5, 6],
    '5': [6],
    '6': [7],
    '7': ['out']
    }

    # Optuna Hyperparameter Study
    study_name = parser_args.study_name
    search_space = {
        'batch_size': [32],
        'ret_channels': [32],
        'dropout': [0.0],
        'model_class': ["RetinaVVSGraph"]
    }
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.GridSampler(search_space))
    search_space["vvs_graph"] = vvs_graph
    study.optimize(lambda trials: objective(trials, parser_args, search_space), n_trials=parser_args.n_trials)

    # Process study dataframe
    study_df = study.trials_dataframe()
    study_df.rename(columns={"value": "val_acc", "number": "trial"}, inplace=True)
    study_df.drop(["datetime_start", "datetime_complete"], axis=1, inplace=True)
    study_df.to_hdf(f"RetinaVVSGraph/studies/{study_name}.h5", key="study")
