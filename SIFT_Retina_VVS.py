from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import loggers as pl_loggers
from kornia.feature.siftdesc import SIFTDescriptor
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


class SIFT(nn.Module):
    """
    A SIFT class that automatically processes one batch of images using kornia's sift descriptor
    """

    def __init__(self, patch_size=65, num_ang_bins=8, num_spatial_bins=4, clip_val=0.2, root_sift=False):
        super(SIFT, self).__init__()
        self.ps = patch_size
        self.sift = SIFTDescriptor(patch_size, num_ang_bins, num_spatial_bins, root_sift, clip_val)

    def forward(self, image_batch):
        bs = image_batch.shape[0]  # Batch size
        c = image_batch.shape[1]  # Number of image channels
        ps = self.ps  # Patch size

        # Get SIFT outputs
        patches = image_batch.unfold(2, ps, ps).unfold(3, ps, ps).reshape(-1, c, ps, ps)
        p_c = torch.split(patches, 1, dim=1)
        outputs = torch.stack([self.sift(image) for image in p_c]).permute(1, 0, 2)

        return outputs


class SIFTRetinaVVS(pl.LightningModule):

    def __init__(self, batch_size, input_shape, ret_channels, vvs_layers, patch_size, dropout):
        super(SIFTRetinaVVS, self).__init__()

        # Model Parameters
        self.batch_size = batch_size
        self.name = f"RetChannels-{ret_channels}_VVSLayers-{vvs_layers}_PatchSize-{patch_size}_"
        self.name += f"Dropout-{int(dropout * 100)}_BatchSize-{batch_size}"

        # Retina Net
        self.inputs = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=9)
        self.ret_bn1 = nn.BatchNorm2d(num_features=32)
        self.ret_conv = nn.Conv2d(in_channels=32, out_channels=ret_channels, kernel_size=9)
        self.ret_bn2 = nn.BatchNorm2d(num_features=ret_channels)

        # VVS_Net
        self.vvs_conv = nn.ModuleList()
        self.vvs_bn = nn.ModuleList()
        self.vvs_conv.append(nn.Conv2d(in_channels=ret_channels, out_channels=32, kernel_size=9))
        self.vvs_bn.append(nn.BatchNorm2d(num_features=32))
        for _ in range(vvs_layers-1):
            self.vvs_conv.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=9))
            self.vvs_bn.append(nn.BatchNorm2d(num_features=32))
        self.sift = SIFT(patch_size=patch_size)
        in_features = 32 * input_shape[0] * int(input_shape[1]/patch_size)**2 * 128  # Number of SIFT outputs
        self.vvs_fc = nn.Linear(in_features=in_features, out_features=1024)
        self.outputs = nn.Linear(in_features=1024, out_features=10)

        # Define Dropout and Padding Layers
        self.pad = nn.ZeroPad2d(4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, t):
        batch_size = t.shape[0]

        # Retina forward pass
        t = self.pad(self.ret_bn1(F.relu(self.inputs(t))))
        t = self.pad(self.ret_bn2(F.relu(self.ret_conv(t))))

        # VVS forward pass
        for conv, bn in zip(self.vvs_conv, self.vvs_bn):
            t = self.pad(bn(F.relu(conv(t))))
        t = self.sift(t).reshape(batch_size, -1)
        t = self.dropout(F.relu(self.vvs_fc(t)))
        t = self.outputs(t)

        return t

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
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
            "time": time.time() - start
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
        return self.training_step(batch, batch_id)

    def validation_epoch_end(self, outputs):
        # Get ROC_AUC
        labels = np.concatenate([batch["labels"].cpu().numpy() for batch in outputs])
        predictions = np.concatenate([batch["predictions"].cpu().numpy() for batch in outputs])
        auc = roc_auc_score(labels, predictions, multi_class="ovr")

        # Get epoch average metrics
        avg_loss = torch.stack([batch["loss"] for batch in outputs]).mean()
        avg_acc = np.stack([batch["acc"] for batch in outputs]).mean()
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
    ret_channels = trial.suggest_categorical("ret_channels", [8, 16, 32, 64])
    vvs_layers = trial.suggest_int("vvs_layers", 1, 6)
    patch_size = trial.suggest_categorical("patch_size", [8, 16, 32])
    dropout = trial.suggest_discrete_uniform("dropout", 0.0, 0.5, 0.01)

    # Train and validation dataloaders
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor()
    ])
    train_data = CIFAR10(root="data/CIFAR10", download=False, train=True, transform=transform)
    val_size = int(0.2 * len(train_data))
    train_size = len(train_data) - val_size
    train_data, val_data = random_split(train_data, [train_size, val_size],
                                        generator=torch.Generator().manual_seed(trial.number))
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=12)
    val_data = DataLoader(val_data, batch_size=batch_size, num_workers=12)

    # Create model
    model = SIFTRetinaVVS(batch_size, (1, 32, 32), ret_channels,
                          vvs_layers, patch_size, dropout)

    # Callbacks
    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min"
    )
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        filepath="models/SIFT/",
        prefix=f"trial_{trial.number}",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )
    tb_logger = pl_loggers.TensorBoardLogger("logs/SIFT/", name=f"trial_{trial.number}")

    # Train the model
    trainer = pl.Trainer(gpus=1, early_stop_callback=early_stop,
                         checkpoint_callback=model_checkpoint, max_epochs=100,
                         fast_dev_run=False, logger=tb_logger)
    trainer.fit(model, train_dataloader=train_data, val_dataloaders=val_data)

    # Evaluate model
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
    parser.add_argument("--n_trials", default=1)
    args = parser.parse_args()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)
    study.trials_dataframe().to_hdf("studies/SIFT", key="study")
