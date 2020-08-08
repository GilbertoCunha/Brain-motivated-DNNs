from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from argparse import ArgumentParser
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np
import torch
import time


class RetinaVVSNet(pl.LightningModule):

    def __init__(self, batch_size, input_shape, ret_channels, vvs_layers, dropout):
        super(RetinaVVSNet, self).__init__()
        
        # Model Parameters
        self.batch_size = batch_size
        self.name = f"RetChannels-{ret_channels}_VVSLayers-{vvs_layers}_Dropout-{dropout}_"
        self.name += f"BatchSize-{batch_size}"

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

    def prepare_data(self):
        CIFAR10("data/CIFAR10", train=True, download=True)
        CIFAR10("data/CIFAR10", train=False, download=True)

    def train_dataloader(self):
        # Select transformations for the data
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

        # Split data and create training DataLoader
        train_data = CIFAR10("data/CIFAR10", train=True,
                             download=False, transform=transform)
        train_data = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        return train_data

    def val_dataloader(self):
        # Select transformations for the data
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
        val_data = CIFAR10("data/CIFAR10", train=False,
                           download=False, transform=transform)
        val_data = DataLoader(val_data, batch_size=self.batch_size)
        
        return val_data

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def cross_entropy_loss(self, predictions, labels):
        return F.cross_entropy(predictions, labels)

    def training_step(self, batch, batch_id):
        start = time.time()

        # Get predictions
        images, labels = batch
        predictions = self.forward(images)

        # Get batch metrics
        accuracy = accuracy_score(
            labels.cpu().detach().numpy(),
            predictions.argmax(dim=-1).cpu().detach().numpy()
        )
        loss = self.cross_entropy_loss(predictions, labels)

        r = {"labels": labels, "predictions": F.softmax(predictions, dim=-1),
             "loss": loss, "acc": accuracy, "time": time.time()-start}

        return r

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
        tensorboard_logs = {"train_loss": avg_loss, "train_acc": avg_acc,
                            "train_auc": auc, "epoch_duration": total_time}

        return {"avg_train_loss": avg_loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_id):
        start = time.time()

        # Get predictions
        images, labels = batch
        predictions = self.forward(images)

        accuracy = accuracy_score(
            labels.cpu(),
            predictions.argmax(dim=-1).cpu()
        )

        # Get batch metrics
        loss = self.cross_entropy_loss(predictions, labels)

        r = {"labels": labels, "predictions": F.softmax(predictions, dim=-1),
             "val_loss": loss, "val_acc": accuracy, "time": time.time()-start}

        return r

    def validation_epoch_end(self, outputs):
        # Get epoch average metrics
        avg_loss = torch.stack([batch["val_loss"] for batch in outputs]).mean()
        avg_acc = np.stack([batch["val_acc"] for batch in outputs]).mean()
        total_time = np.stack([batch["time"] for batch in outputs]).sum()

        # Get ROC_AUC
        labels = np.concatenate([batch["labels"].cpu().numpy() for batch in outputs])
        predictions = np.concatenate([batch["predictions"].cpu().numpy() for batch in outputs])
        auc = roc_auc_score(labels, predictions, multi_class="ovr")

        # Tensorboard log
        tensorboard_logs = {"val_loss": avg_loss, "val_acc": avg_acc,
                            "val_auc": auc, "epoch_duration": total_time}

        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}


if __name__ == "__main__":
    # Terminal Arguments
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--input_shape", default=(1, 32, 32))
    parser.add_argument("--ret_channels", default=32)
    parser.add_argument("--vvs_layers", default=4)
    parser.add_argument("--dropout", default=0.4, type=float)
    parser.add_argument("--ES_patience", default=3)
    parser.add_argument("--save_top_k", default=1)
    args = parser.parse_args()

    # Create model
    model = RetinaVVSNet(args.batch_size, args.input_shape, args.ret_channels,
                         args.vvs_layers, args.dropout)

    # Callbacks
    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=args.ES_patience,
        mode="min"
    )
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        filepath="lightning_models/" + model.name,
        monitor="val_loss",
        mode="min",
        save_top_k=args.save_top_k
    )

    # Train the model
    trainer = pl.Trainer.from_argparse_args(args, gpus=1, early_stop_callback=early_stop,
                                            checkpoint_callback=model_checkpoint, max_epochs=100)
    trainer.fit(model)
