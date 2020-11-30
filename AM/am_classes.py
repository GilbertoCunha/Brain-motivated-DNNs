from RetinaVVS.RetinaVVS_class import RetinaVVS
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torch


class ChannelAttention (pl.LightningModule):
    def __init__(self, channels, kernel_size):
        # Kernel size should be the dimensions of the square image
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward (self, t):
        t = self.avg_pool(t)
        t = self.conv(t)
        t = nn.Softmax2d()(t)
        return t
    
    
class SpacialAttention (pl.LightningModule):
    def __init__(self, channels):
        super(SpacialAttention, self).__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        
    def forward (self, t):
        t = self.conv(t)
        t = t / torch.sum(t)
        return t
    

class AttentionBlock(pl.LightningModule):
    def __init__(self, channels, kernel_size):
        super(AttentionBlock, self).__init__()
        self.SA = SpacialAttention(channels)
        self.CA = ChannelAttention(channels, kernel_size)
        
    def forward (self, t):
        channel_weights = self.CA(t)
        pixel_weights = self.SA(t)
        return t * pixel_weights * channel_weights
    
    
class AMRetinaStart(RetinaVVS):
    def __init__(self, hparams):
        super(AMRetinaStart, self).__init__(hparams)
        self.spacial_attention = SpacialAttention(self.ret_channels)
        self.channel_attention = ChannelAttention(self.ret_channels)

    def forward(self, tensor):
        # Retina forward pass
        t = self.pad(self.ret_bn1(F.relu(self.inputs(tensor))))
        t = self.pad(self.ret_bn2(F.relu(self.ret_conv(t))))
        
        # Apply spacial and channel attention
        channel_weights = self.channel_attention(t)
        pixel_weights = self.spacial_attention(t)
        t = t * channel_weights * pixel_weights

        # VVS forward pass
        for conv, bn in zip(self.vvs_conv, self.vvs_bn):
            t = self.pad(bn(F.relu(conv(t))))
        t = self.dropout(F.relu(self.vvs_fc(t)))
        t = self.outputs(t)

        return t

    
if __name__ == '__main__':
    tensor = torch.randn([32, 32, 32, 32])
    Attention = AttentionBlock(32, 32)
    print (Attention(tensor).shape)