from RetinaVVS.RetinaVVS_class import RetinaVVS
import torch.nn.functional as F
import torch.nn as nn
import torch


class ConvLBP(nn.Conv2d):
    """
    Class taken from dizcza repo:
    https://github.com/dizcza/lbcnn.pytorch/blob/master/lbcnn_model.py
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, sparsity=0.5):
        super().__init__(in_channels, out_channels, kernel_size, padding=1, bias=False)
        weights = next(self.parameters())
        matrix_proba = torch.FloatTensor(weights.data.shape).fill_(0.5)
        binary_weights = torch.bernoulli(matrix_proba) * 2 - 1
        mask_inactive = torch.rand(matrix_proba.shape) > sparsity
        binary_weights.masked_fill_(mask_inactive, 0)
        weights.data = binary_weights
        weights.requires_grad_(False)


class LBPRetinaStart(RetinaVVS):
    def __init__(self, hparams):
        super(LBPRetinaStart, self).__init__(hparams)

        # Gather hparams
        input_shape = hparams["input_shape"]
        out_channels = hparams["out_channels"]
        kernel_size = hparams["kernel_size"]
        sparsity = hparams["sparsity"]

        # Model identifiers
        self.name += f"_OutChans{out_channels}_Kernel{kernel_size}_Spars{sparsity}"

        # Modify model parameters
        self.lbp = ConvLBP(in_channels=1, out_channels=out_channels, kernel_size=kernel_size, sparsity=sparsity)
        # TODO: generalize number of features
        # in_features = (32 + out_channels) * input_shape[1] * input_shape[2]
        in_features = int(out_channels * 36992 / 32 + 32 * input_shape[1] * input_shape[2])
        self.vvs_fc = nn.Linear(in_features=in_features, out_features=1024)

    def forward(self, tensor):
        batch_size = tensor.shape[0]

        # Retina forward pass
        t_ = self.pad(F.relu(self.lbp(tensor))).reshape(batch_size, -1)
        t = self.pad(self.ret_bn1(F.relu(self.inputs(tensor))))
        t = self.pad(self.ret_bn2(F.relu(self.ret_conv(t))))

        # VVS forward pass
        for conv, bn in zip(self.vvs_conv, self.vvs_bn):
            t = self.pad(bn(F.relu(conv(t))))
        t = torch.cat((t.reshape(batch_size, -1), t_), dim=-1)
        t = self.dropout(F.relu(self.vvs_fc(t)))
        t = self.outputs(t)

        return t


class LBPVVSEnd(RetinaVVS):
    def __init__(self, hparams):
        super(LBPVVSEnd, self).__init__(hparams)

        # Gather hparams
        input_shape = hparams["input_shape"]
        out_channels = hparams["out_channels"]
        kernel_size = hparams["kernel_size"]
        sparsity = hparams["sparsity"]

        # Model identifiers
        self.name += f"_OutChans{out_channels}_Kernel{kernel_size}_Spars{sparsity}"

        # Change model parameters
        self.lbp = ConvLBP(in_channels=32, out_channels=out_channels, kernel_size=kernel_size, sparsity=sparsity)
        # TODO: generalize number of features
        in_features = int(out_channels * 36992 / 32)
        self.vvs_fc = nn.Linear(in_features=in_features, out_features=1024)

    def forward(self, tensor):
        batch_size = tensor.shape[0]

        # Retina forward pass
        t = self.pad(self.ret_bn1(F.relu(self.inputs(tensor))))
        t = self.pad(self.ret_bn2(F.relu(self.ret_conv(t))))

        # VVS forward pass
        for conv, bn in zip(self.vvs_conv, self.vvs_bn):
            t = self.pad(bn(F.relu(conv(t))))
        t = self.pad(F.relu(self.lbp(t))).reshape(batch_size, -1)
        t = self.dropout(F.relu(self.vvs_fc(t)))
        t = self.outputs(t)

        return t


class LBPBoth(RetinaVVS):
    def __init__(self, hparams):
        super(LBPBoth, self).__init__(hparams)

        # Gather hparams
        input_shape = hparams["input_shape"]
        out_channels = hparams["out_channels"]
        kernel_size = hparams["kernel_size"]
        sparsity = hparams["sparsity"]

        # Model Parameters
        self.name += f"_OutChans{out_channels}_Kernel{kernel_size}_Spars{sparsity}"

        # Modify model parameters
        self.lbp_start = ConvLBP(in_channels=1, out_channels=out_channels, kernel_size=kernel_size, sparsity=sparsity)
        self.lbp_end = ConvLBP(in_channels=32, out_channels=out_channels, kernel_size=kernel_size, sparsity=sparsity)
        # TODO: generalize number of features
        in_features = int(2 * out_channels * 36992 / 32)
        self.vvs_fc = nn.Linear(in_features=in_features, out_features=1024)

    def forward(self, t):
        batch_size = t.shape[0]

        # Retina forward pass
        t_ = self.pad(F.relu(self.lbp_start(t))).reshape(batch_size, -1)
        t = self.pad(self.ret_bn1(F.relu(self.inputs(t))))
        t = self.pad(self.ret_bn2(F.relu(self.ret_conv(t))))

        # VVS forward pass
        for conv, bn in zip(self.vvs_conv, self.vvs_bn):
            t = self.pad(bn(F.relu(conv(t))))
        t = self.pad(F.relu(self.lbp_end(t))).reshape(batch_size, -1)
        t = torch.cat((t.reshape(batch_size, -1), t_), dim=-1)
        t = self.dropout(F.relu(self.vvs_fc(t)))
        t = self.outputs(t)

        return t
