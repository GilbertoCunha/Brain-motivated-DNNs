from sklearn.metrics import roc_auc_score
import SIFT.SIFT_classes as SIFT_classes
import torch.nn.functional as F
import pytorch_lightning as pl
from pathlib import Path
import torch.nn as nn
import numpy as np
import torch
import time
import json


class RetinaVVS(pl.LightningModule):
    def __init__(self, hparams):
        self.hparams = hparams
        super(RetinaVVS, self).__init__()
        self.avg_acc = []

        # Gather hparams
        input_shape = hparams["input_shape"]
        ret_channels = hparams["ret_channels"]
        vvs_layers = hparams["vvs_layers"]
        dropout = hparams["dropout"]
        self.ret_channels = ret_channels
        self.vvs_layers = vvs_layers
        self.drop = dropout

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
        
        # Log to tensorboard
        self.log("train_loss", avg_loss)
        self.log("train_acc", avg_acc)
        self.log("epoch_duration", total_time)
        self.log("step", self.current_epoch)
        
        if self.current_epoch != 0:
            # Get ROC_AUC
            labels = np.concatenate([batch["labels"].detach().cpu().numpy() for batch in outputs])
            predictions = np.concatenate([batch["predictions"].detach().cpu().numpy() for batch in outputs])
            auc = roc_auc_score(labels, predictions, multi_class="ovr")
            self.log("train_auc", auc)
        else:
            if "SIFT" in self.hparams["model_class"]:
                model = getattr(SIFT_classes, self.hparams["model_class"])(self.hparams)
            else:
                model = RetinaVVS()
            # Log graph to tensorboard
            sampleImg=torch.rand((1, 1, 32, 32))
            self.logger.experiment.add_graph(model, sampleImg)

    def validation_step(self, batch, batch_id):
        return self.training_step(batch, batch_id)

    def validation_epoch_end(self, outputs):
        # Get epoch average metrics
        avg_loss = torch.stack([batch["loss"] for batch in outputs]).mean()
        avg_acc = torch.stack([batch["acc"] for batch in outputs]).mean()
        total_time = np.stack([batch["time"] for batch in outputs]).sum()
        
        # Log to tensorboard and prog_bar
        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_acc", avg_acc, prog_bar=True)
        self.log("total_time", total_time)

        if self.current_epoch != 0:
            # Get ROC_AUC
            labels = np.concatenate([batch["labels"].detach().cpu().numpy() for batch in outputs])
            predictions = np.concatenate([batch["predictions"].detach().cpu().numpy() for batch in outputs])
            auc = roc_auc_score(labels, predictions, multi_class="ovr")
            self.log("val_auc", auc, prog_bar=True)
            
            # Save best model
            self.avg_acc.append(avg_acc)
            if avg_acc >= max(self.avg_acc):
                Path(f"Best_Models/{self.filename}/{self.name}").mkdir(parents=True, exist_ok=True)
                torch.save(self.state_dict(), f"Best_Models/{self.filename}/{self.name}/weights.tar")
                
                with open(f"Best_Models/{self.filename}/{self.name}/parameters.txt", "w") as file:
                    json.dump(self.hparams, file)
                    
                file.close()
