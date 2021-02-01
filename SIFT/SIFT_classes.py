import RetinaVVS.RetinaVVS_class as RetinaVVS_class
from kornia.feature.siftdesc import SIFTDescriptor
import torch.nn.functional as F
import torch.nn as nn
import torch


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
    
    
class SIFTRetinaStart(RetinaVVS_class.RetinaVVS):
    def __init__(self, hparams):
        super(SIFTRetinaStart, self).__init__(hparams)

        # Gather hparams
        input_shape = hparams["input_shape"]
        patch_size = hparams["patch_size"]
        ret_channels = hparams["ret_channels"]
        self.patch_size = patch_size

        # Model identifiers
        self.filename = "SIFTRetinaStart"
        self.name += f"_PatchSize{patch_size}"

        # Modify model parameters
        vvs_features = ret_channels * input_shape[1] * input_shape[2]
        sift_features = 128 * ret_channels * int(input_shape[1] / patch_size) ** 2
        self.vvs_fc = nn.Linear(in_features=vvs_features+sift_features, out_features=(vvs_features+sift_features)//128)
        self.outputs = nn.Linear(in_features=(vvs_features+sift_features)//128, out_features=10)
        self.sift = SIFT(patch_size=patch_size)

    def forward(self, tensor):
        batch_size = tensor.shape[0]

        # Retina forward pass
        t = self.pad(self.ret_bn1(F.relu(self.inputs(tensor))))
        t = self.pad(self.ret_bn2(F.relu(self.ret_conv(t))))
        
        # Apply sift after retina
        t_sift = self.sift(t).reshape(batch_size, -1)

        # VVS forward pass
        for conv, bn in zip(self.vvs_conv, self.vvs_bn):
            t = self.pad(bn(F.relu(conv(t))))
            
        # Fully connected network
        t = torch.cat((t.reshape(batch_size, -1), t_sift), dim=-1)
        t = self.dropout(F.relu(self.vvs_fc(t)))
        t = self.outputs(t)

        return t


class SIFTVVSEnd(RetinaVVS_class.RetinaVVS):
    def __init__(self, hparams):
        super(SIFTVVSEnd, self).__init__(hparams)

        # Gather hparams
        input_shape = hparams["input_shape"]
        patch_size = hparams["patch_size"]
        self.patch_size = patch_size

        # Model identifiers
        self.filename = "SIFTVVSEnd"
        self.name += f"_PatchSize{patch_size}"

        # Modify model parameters
        sift_features = 128 * input_shape[1] * int(input_shape[1] / patch_size) ** 2
        self.vvs_fc = nn.Linear(in_features=sift_features, out_features=1024)
        self.sift = SIFT(patch_size=patch_size)

    def forward(self, tensor):
        batch_size = tensor.shape[0]

        # Retina forward pass
        t = self.pad(self.ret_bn1(F.relu(self.inputs(tensor))))
        t = self.pad(self.ret_bn2(F.relu(self.ret_conv(t))))

        # VVS forward pass
        for conv, bn in zip(self.vvs_conv, self.vvs_bn):
            t = self.pad(bn(F.relu(conv(t))))
            
        # Apply SIFT
        t = self.sift(t).reshape(batch_size, -1)
        
        # Fully Connected Network
        t = self.dropout(F.relu(self.vvs_fc(t)))
        t = self.outputs(t)

        return t


class SIFTBoth(RetinaVVS_class.RetinaVVS):
    def __init__(self, hparams):
        super(SIFTBoth, self).__init__(hparams)

        # Gather hparams
        ret_channels = hparams["ret_channels"]
        input_shape = hparams["input_shape"]
        patch_size = hparams["patch_size"]
        self.patch_size = patch_size

        # Model Parameters
        self.filename = "SIFTBoth"
        self.name += f"_PatchSize{patch_size}"

        # Modify model parameters
        vvs_features = 128 * input_shape[1] * int(input_shape[1] / patch_size) ** 2
        sift_features = 128 * ret_channels * int(input_shape[1] / patch_size) ** 2
        self.vvs_fc = nn.Linear(in_features=vvs_features+sift_features, out_features=(vvs_features+sift_features)//128)
        self.outputs = nn.Linear(in_features=(vvs_features+sift_features)//128, out_features=10)
        self.sift = SIFT(patch_size=patch_size)

    def forward(self, tensor):
        batch_size = tensor.shape[0]

        # Retina forward pass
        t = self.pad(self.ret_bn1(F.relu(self.inputs(tensor))))
        t = self.pad(self.ret_bn2(F.relu(self.ret_conv(t))))
        
        # Apply sift after retina
        sift_retina = self.sift(t).reshape(batch_size, -1)

        # VVS forward pass
        for conv, bn in zip(self.vvs_conv, self.vvs_bn):
            t = self.pad(bn(F.relu(conv(t))))
            
        # Apply sift after VVS
        sift_vvs = self.sift(t).reshape(batch_size, -1)
        
        # Fully Connected Network
        t = torch.cat((sift_vvs, sift_retina), dim=-1)
        t = self.dropout(F.relu(self.vvs_fc(t)))
        t = self.outputs(t)

        return t