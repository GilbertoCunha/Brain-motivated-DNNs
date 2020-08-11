from sklearn.metrics import accuracy_score, roc_auc_score
from pytorch_lightning import loggers as pl_loggers
from kornia.feature.siftdesc import SIFTDescriptor
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from torchvision import transforms
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
        train_data = DataLoader(train_data, batch_size=self.batch_size,
                                shuffle=True, num_workers=12)

        return train_data

    def val_dataloader(self):
        # Select transformations for the data
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
        val_data = CIFAR10("data/CIFAR10", train=False,
                           download=False, transform=transform)
        val_data = DataLoader(val_data, batch_size=self.batch_size, num_workers=12)

        return val_data

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


if __name__ == "__main__":
    # Terminal Arguments
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--input_shape", default=(1, 32, 32))
    parser.add_argument("--ret_channels", default=32)
    parser.add_argument("--vvs_layers", default=4)
    parser.add_argument("--patch_size", default=16)
    parser.add_argument("--dropout", default=0.4, type=float)
    parser.add_argument("--ES_patience", default=10)
    parser.add_argument("--save_top_k", default=1)
    args = parser.parse_args()

    # Create model
    model = SIFTRetinaVVS(args.batch_size, args.input_shape, args.ret_channels,
                          args.vvs_layers, args.patch_size, args.dropout)

    # Callbacks
    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=args.ES_patience,
        mode="min"
    )
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        filepath="models/SIFT/",
        prefix=model.name,
        monitor="val_loss",
        mode="min",
        save_top_k=args.save_top_k
    )
    tb_logger = pl_loggers.TensorBoardLogger("logs/SIFT/", name=model.name)

    # Train the model
    trainer = pl.Trainer.from_argparse_args(args, gpus=1, early_stop_callback=early_stop,
                                            checkpoint_callback=model_checkpoint, max_epochs=100,
                                            fast_dev_run=False, logger=tb_logger)
    trainer.fit(model)
