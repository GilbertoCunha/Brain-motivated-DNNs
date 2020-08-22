from kornia.feature.siftdesc import SIFTDescriptor
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np
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
        c = image_batch.shape[1]  # Number of image channels
        ps = self.ps  # Patch size

        # Get SIFT outputs
        patches = image_batch.unfold(2, ps, ps).unfold(3, ps, ps).reshape(-1, c, ps, ps)
        p_c = torch.split(patches, 1, dim=1)
        outputs = torch.stack([self.sift(image) for image in p_c]).permute(1, 0, 2)

        return outputs


class SIFTRetinaVVS(pl.LightningModule):

    def __init__(self, hparams):
        super(SIFTRetinaVVS, self).__init__()

        # Gather model hparams
        input_shape = hparams["input_shape"]
        ret_channels = hparams["ret_channels"]
        vvs_layers = hparams["vvs_layers"]
        dropout = hparams["dropout"]
        if hparams["sift_type"] != "none":
            patch_size = hparams["patch_size"]

        # Model Parameters
        self.sift_type = hparams["sift_type"]
        self.lr = hparams["lr"]
        self.name = f"RetinaChans{ret_channels}_VVSLayers{vvs_layers}"
        if hparams["sift_type"] != "none":
            self.name += f"_PatchSize{patch_size}"

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
        if self.sift_type == "ret_start":
            in_features = 32 * 32 * 32 + 128 * input_shape[0] * int(input_shape[1] / patch_size) ** 2
        elif self.sift_type == "vvs_end":
            in_features = 32 * input_shape[0] * int(input_shape[1] / patch_size) ** 2 * 128
        elif self.sift_type == "both":
            in_features = 33 * input_shape[0] * int(input_shape[1] / patch_size) ** 2 * 128
        else:
            in_features = 32 * 32 * 32

        self.vvs_fc = nn.Linear(in_features=in_features, out_features=1024)
        self.outputs = nn.Linear(in_features=1024, out_features=10)

        # Define Dropout, Padding and SIFT Layers
        if self.sift_type != "none":
            self.sift = SIFT(patch_size=patch_size)
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
        if self.sift_type == "vvs_end":
            t = self.sift(t).reshape(batch_size, -1)
        elif self.sift_type == "ret_start":
            t = torch.cat((t.reshape(batch_size, -1), self.sift(tensor).reshape(batch_size, -1)), dim=-1)
        elif self.sift_type == "both":
            t = self.sift(t).reshape(batch_size, -1)
            t = torch.cat((t.reshape(batch_size, -1), self.sift(tensor).reshape(batch_size, -1)), dim=-1)
        else:
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

        return results
