from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np
import torch
import time


def channels_graph(g, r_c):
    """
    This function calculates a dictionary of the number of channels each layer
    of the VVS-Network receives according to any connection network determined by 
    the graph 'g'. It assumes that the number of output channels in any layer is 
    always equal to the number of input channels.

    Args:
        g (dict): A graph that describes the connections to be made between layers.
        It should be constructed the following way:
            1 - The keys are 'str(i)' where 'i' is the layer number
            2 - The key "out" corresponds to the output of the vvs_network
            3 - The values 'g[str(i)]' are lists '[j]' where the layer 'g[str(j)]'
            has a connection coming from layer 'g[str(i)]'
        r_c (int): The number of channels that leave the Retina Network

    Returns:
        dict: dictionary with the connections and the number of input channels
        for each graph. The output graph keys are the same, but its values are
        now tuples '(a, [b])' such that:
            - g[key][0]=a are the number of channels that enter the layer 'key'
            - g[key][2]=[b] are all the layers 'b' that connect to channel 'key'
    """
    
    # Calculate the channels for each layer
    r, values = {}, [r_c]
    for i in range(1, len(list(g.keys()))):
        indexes = [int(key) for key in g if i in g[key]]
        r[str(i)] = (sum([values[j] for j in indexes]), indexes)
        values.append(r[str(i)][0])
        
    # Calculate the output channels and connections
    indexes = [int(key) for key in g if "out" in g[key]]
    r["out"] = (sum([values[j] for j in indexes]), indexes)
        
    return r


class RetinaVVSGraph(pl.LightningModule):
    def __init__(self, hparams):
        super(RetinaVVSGraph, self).__init__()

        # Gather hparams
        self.save_hyperparameters(hparams)
        input_shape = hparams["input_shape"]
        ret_channels = hparams["ret_channels"]
        vvs_graph = hparams["vvs_graph"]
        dropout = hparams["dropout"]
        self.lr = hparams["lr"]
        self.filename = "RetinaVVSGraph"
        self.vvs_graph = vvs_graph
        self.dropout = dropout
        self.ret_channels = ret_channels
        
        # Model name
        self.graph = channels_graph(vvs_graph, ret_channels)
        self.name = f"RetChans{ret_channels}_Graph{vvs_graph}"

        # Retina Net
        self.inputs = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=9)
        self.ret_bn1 = nn.BatchNorm2d(num_features=32)
        self.ret_conv = nn.Conv2d(in_channels=32, out_channels=ret_channels, kernel_size=9)
        self.ret_bn2 = nn.BatchNorm2d(num_features=ret_channels)

        # VVS_Net
        self.vvs_conv = nn.ModuleList()
        self.vvs_bn = nn.ModuleList()
        self.vvs_1x1 = nn.ModuleDict()
        print(list(self.graph.keys()))
        for key in self.graph:
            if key != "out":
                num_channels = self.graph[key][0]
                self.vvs_conv.append(nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=9))
                self.vvs_bn.append(nn.BatchNorm2d(num_features=num_channels))
        
                for value in self.graph[key][1]:
                    if value == 0:
                        # Possible mistake in number of channels. Check if error arises
                        num_channels = ret_channels
                    else:
                        num_channels = self.graph[str(value)][0]
                    self.vvs_1x1[f"{value},{key}"] = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=1)
        
        features = self.graph["out"][0] * input_shape[1] * input_shape[2]
        
        # NOTE: This neural network might need more complexity and
        # layers due to the huge ammount of input_features
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
        # TODO: Make the graph allow output_channels != input_channels
        t_layer_out = [t]
        for key, conv, bn in zip(self.graph, self.vvs_conv, self.vvs_bn):
            # NOTE: this cycle doesn't pass through the "out" graph node
            # because of the zip function and different argument len
            # t_list = [t_layer_out[j] for j in self.graph[key][1]]
            t_list = []
            for j in self.graph[key][1]:
                t_list.append(self.vvs_1x1[f"{j},{key}"](t_layer_out[j]))
            t = torch.cat(t_list, dim=1)
            t = self.pad(bn(F.relu(conv(t))))
            t_layer_out.append(t)
        t = torch.cat([t_layer_out[j] for j in self.graph["out"][1]], dim=1)
        t = t.reshape(batch_size, -1)
        t = self.dropout(F.relu(self.vvs_fc(t)))
        t = self.outputs(t)

        return t

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    @staticmethod
    def cross_entropy_loss(predictions, labels):
        r = F.cross_entropy(predictions, labels)
        return r

    def training_step(self, batch, new):
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
            # Log graph to tensorboard
            model = RetinaVVSGraph(self.hparams)
            sampleImg=torch.rand((1, 1, 32, 32))
            self.logger.experiment.add_graph(model, sampleImg)

    def validation_step(self, batch, batch_id):
        return self.training_step(batch, batch_id)

    def validation_epoch_end(self, outputs):
        # Get epoch average metrics
        avg_loss = torch.stack([batch["loss"] for batch in outputs]).mean()
        avg_acc = torch.stack([batch["acc"] for batch in outputs]).mean()
        
        # Log to tensorboard and prog_bar
        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_acc", avg_acc, prog_bar=True)
