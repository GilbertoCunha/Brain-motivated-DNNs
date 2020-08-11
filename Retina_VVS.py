from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import loggers as pl_loggers
from torchvision.datasets import CIFAR10
from argparse import ArgumentParser
from torchvision import transforms
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import numpy as np
import optuna
import torch
import time


class RetinaVVSNet(pl.LightningModule):
    """
    This is the network class
    This network is a combination of two networks with different purposes:
        - The Retina Net: Captures and encodes visual data
        - The VVS Net: Processes the visual data

    Retina Net:
        - 2 convolutional layers
        - 2 regulatory batch normalization layers
        - Variable number of output channels to simulate amount of information
        bottleneck to be passed to the VVS Net

    VVS Net:
        - Variable number of convolutional layers to simulate different visual
        cortex complexities
        - Regulatory Batch Normalization at each convolutional layer
        - Fully connected layers to predict image classes
    """

    def __init__(self, batch_size, input_shape, ret_channels, vvs_layers, dropout):
        super(RetinaVVSNet, self).__init__()
        
        # Model Parameters
        self.batch_size = batch_size
        self.name = f"RetChannels-{ret_channels}_VVSLayers-{vvs_layers}_Dropout-{int(dropout*100)}_"
        self.name += f"BatchSize-{batch_size}_Optim:RMSprop"

        # Define Retina Net
        self.ret_conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=9)
        self.ret_bn1 = nn.BatchNorm2d(num_features=32)
        self.ret_out = nn.Conv2d(in_channels=32, out_channels=ret_channels, kernel_size=9)
        self.ret_bn2 = nn.BatchNorm2d(num_features=ret_channels)

        # Define VVS_Net
        self.vvs_conv = nn.ModuleList()
        self.vvs_bn = nn.ModuleList()
        self.vvs_conv.append(nn.Conv2d(in_channels=ret_channels, out_channels=32, kernel_size=9))
        self.vvs_bn.append(nn.BatchNorm2d(num_features=32))
        for _ in range(vvs_layers - 1):
            self.vvs_conv.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=9))
            self.vvs_bn.append(nn.BatchNorm2d(num_features=32))
        self.vvs_fc = nn.Linear(in_features=input_shape[1]*input_shape[2]*32, out_features=1024)
        self.vvs_out = nn.Linear(in_features=1024, out_features=10)

        # Padding and Dropout layers
        self.pad = nn.ZeroPad2d(4)
        self.dropout = nn.Dropout(dropout)

    def __repr__(self):
        return self.name

    def forward(self, t):
        # Retina forward pass
        t = self.pad(self.ret_bn1(F.relu(self.ret_conv1(t))))
        t = self.pad(self.ret_bn2(F.relu(self.ret_out(t))))

        # VVS forward pass
        for Conv2d, BatchNorm2d in zip(self.vvs_conv, self.vvs_bn):
            t = self.pad(BatchNorm2d(F.relu(Conv2d(t))))
        t = self.dropout(F.relu(self.vvs_fc(t.view(-1, 32*32*32))))
        t = self.vvs_out(t)
        
        return t

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters())
        return optimizer

    @staticmethod
    def cross_entropy_loss(predictions, labels):
        return F.cross_entropy(predictions, labels)

    def training_step(self, batch, batch_id):
        start = time.time()

        # Get predictions
        images, labels = batch
        predictions = self(images)

        # Get batch metrics
        accuracy = accuracy_score(
            labels.cpu().detach().numpy(),
            predictions.argmax(dim=-1).cpu().detach().numpy()
        )
        loss = self.cross_entropy_loss(predictions, labels)

        # Get train batch output
        output = {
            "labels": labels,
            "predictions": F.softmax(predictions, dim=-1),
            "loss": loss,
            "acc": accuracy,
            "time": time.time()-start
        }

        return output

    def training_epoch_end(self, outputs):
        # Get epoch average metrics
        avg_loss = torch.stack([batch["loss"] for batch in outputs]).mean()
        avg_acc = np.stack([batch["acc"] for batch in outputs]).mean()
        total_time = np.stack([batch["time"] for batch in outputs]).sum()

        # Get ROC_AUC
        labels = np.concatenate([batch["labels"].cpu().numpy() for batch in outputs])
        predictions = np.concatenate([batch["predictions"].cpu().numpy() for batch in outputs])
        auc = roc_auc_score(labels, predictions, multi_class="ovr")

        # Tensorboard log
        tensorboard_logs = {
            "train_loss": avg_loss,
            "train_acc": avg_acc,
            "train_auc": auc,
            "epoch_duration": total_time,
            "step": self.current_epoch
        }

        # Get returns
        results = {
            "train_loss": avg_loss,
            "log": tensorboard_logs
        }

        return results

    def validation_step(self, batch, batch_id):
        start = time.time()

        # Get predictions
        images, labels = batch
        predictions = self(images)

        # Get validation metrics
        accuracy = accuracy_score(
            labels.cpu(),
            predictions.argmax(dim=-1).cpu()
        )
        loss = self.cross_entropy_loss(predictions, labels)

        # Get validation step output
        output = {
            "labels": labels,
            "predictions": F.softmax(predictions, dim=-1),
            "val_loss": loss,
            "val_acc": accuracy,
            "time": time.time()-start
        }

        return output

    def validation_epoch_end(self, outputs):
        # Get ROC_AUC
        labels = np.concatenate([batch["labels"].cpu().numpy() for batch in outputs])
        predictions = np.concatenate([batch["predictions"].cpu().numpy() for batch in outputs])
        auc = roc_auc_score(labels, predictions, multi_class="ovr")

        # Get epoch average metrics
        avg_loss = torch.stack([batch["val_loss"] for batch in outputs]).mean()
        avg_acc = np.stack([batch["val_acc"] for batch in outputs]).mean()
        total_time = np.stack([batch["time"] for batch in outputs]).sum()

        # Progress bar log
        progress_bar = {
            "val_loss": avg_loss,
            "val_acc": avg_acc,
            "val_auc": auc
        }

        # Tensorboard log
        tensorboard_logs = {
            "val_loss": avg_loss,
            "val_acc": avg_acc,
            "val_auc": auc,
            "epoch_duration": total_time,
            "step": self.current_epoch
        }

        # Define return
        results = {
            "val_loss": avg_loss,
            "log": tensorboard_logs,
            "progress_bar": progress_bar
        }

        return results


def objective(trial):
    # Optuna trial parameters
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    ret_channels = trial.suggest_categorical("ret_channels", [4, 8, 16, 32])
    vvs_layers = trial.suggest_int("vvs_layers", 1, 4)
    dropout = trial.suggest_discrete_uniform("dropout", 0.05, 0.5, 0.01)

    # Gather the data
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    train_data = CIFAR10("data/CIFAR10", train=True, download=False, transform=transform)
    val_size = int(0.2 * len(train_data))
    train_size = len(train_data) - val_size
    train_data, val_data = random_split(train_data, [train_size, val_size],
                                        generator=torch.Generator().manual_seed(trial.number))
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=12)
    val_data = DataLoader(val_data, batch_size=batch_size, num_workers=12)

    # Create model
    model = RetinaVVSNet(batch_size, (1, 32, 32), ret_channels,
                         vvs_layers, dropout)

    # Callbacks
    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_acc",
        patience=5,
        mode="max"
    )
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        filepath="models/RetinaVVS/",
        prefix=f"trial_{trial.number}",
        monitor="val_acc",
        mode="max",
        save_top_k=1
    )
    tb_logger = pl_loggers.TensorBoardLogger("logs/RetinaVVS/", name=f"trial_{trial.number}")

    # Train the model
    trainer = pl.Trainer(gpus=1, early_stop_callback=early_stop,
                         checkpoint_callback=model_checkpoint, max_epochs=100,
                         fast_dev_run=True, logger=tb_logger)
    trainer.fit(model, train_dataloader=train_data, val_dataloaders=val_data)

    # Get model accuracy
    val_accuracy = []
    with torch.no_grad():
        for images, labels in tqdm(val_data, total=len(val_data), desc="Evaluating Model"):
            predictions = model(images)
            val_accuracy.append(accuracy_score(labels, predictions.argmax(dim=-1)))
        val_accuracy = sum(val_accuracy) / len(val_accuracy)

    return val_accuracy


if __name__ == "__main__":
    # Terminal Arguments
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--n_trials", default=20)
    args = parser.parse_args()

    # Run optuna bayesian inference hyperparameter search
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)
    study.trials_dataframe().to_hdf("studies/RetinaVVS", key="study")
