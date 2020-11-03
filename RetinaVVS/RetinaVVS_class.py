from sklearn.metrics import roc_auc_score
import pytorch_lightning as pl
import torch.nn.functional as F
from pathlib import Path
import torch.nn as nn
import numpy as np
import torch
import time


class RetinaVVS(pl.LightningModule):
    def __init__(self, hparams):
        super(RetinaVVS, self).__init__()

        # Gather hparams
        input_shape = hparams["input_shape"]
        ret_channels = hparams["ret_channels"]
        vvs_layers = hparams["vvs_layers"]
        dropout = hparams["dropout"]
        self.ret_channels = ret_channels
        self.vvs_layers = vvs_layers
        self.drop = dropout
        self.avg_acc = []

        # Model Parameters
        self.lr = hparams["lr"]
        self.filename = hparams["model_class"]
        self.name = f"RetChans{ret_channels}_VVSLayers{vvs_layers}"

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
        features = 32 * input_shape[1] * input_shape[2]
        self.vvs_fc = nn.Linear(in_features=features, out_features=1024)
        self.outputs = nn.Linear(in_features=1024, out_features=10)

        # Define Dropout, Padding
        self.pad = nn.ZeroPad2d(4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tensor):
        batch_size = tensor.shape[0]

        # Retina forward pass
        t = self.pad(self.ret_bn1(F.relu(self.inputs(tensor))))
        t = self.pad(self.ret_bn2(F.relu(self.ret_conv(t))))

        # VVS forward pass
        for conv, bn in zip(self.vvs_conv, self.vvs_bn):
            t = self.pad(bn(F.relu(conv(t))))
        t = t.reshape(batch_size, -1)
        t = self.dropout(F.relu(self.vvs_fc(t)))
        t = self.outputs(t)

        return t

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
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
        accuracy = predictions.argmax(dim=-1).eq(labels).sum().true_divide(predictions.shape[0])
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
        avg_acc = torch.stack([batch["acc"] for batch in outputs]).mean()
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
        avg_acc = torch.stack([batch["acc"] for batch in outputs]).mean()
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

        # Save best model
        self.avg_acc.append(avg_acc)
        if avg_acc >= max(self.avg_acc):
            Path(f"Best_Models/{self.filename}/{self.name}").mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), f"Best_Models/{self.filename}/{self.name}/weights.tar")
            file = open(f"Best_Models/{self.filename}/{self.name}/parameters.txt", "w")
            if self.filename == "RetinaVVS" or "SIFT" in self.filename:
                file.write(f"Retina Channels: {self.ret_channels}\n")
                file.write(f"VVS Layers: {self.vvs_layers}\n")
                file.write(f"Dropout: {self.drop}\n")
                if "SIFT" in self.filename:
                    file.write(f"Patch Size: {self.patch_size}\n")
                if "LBP" in self.filename:
                    file.write(f"Out Channels: {self.out_channels}")
                    file.write(f"Kernel Size: {self.kernel_size}")
                    file.write(f"Sparsity: {self.sparsity}")
                file.write(f"\nAccuracy: {avg_acc}\n")
                file.write(f"ROC AUC: {auc}\n")
            file.close()

        return results
