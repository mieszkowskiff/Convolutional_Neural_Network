from torchvision import transforms
import kornia.augmentation as K
import torch

import components

class HeadBlock(torch.nn.Module):
    def __init__(self, in_channels = 32, size = 32):
        super(HeadBlock, self).__init__()
        self.head = torch.nn.Sequential(
            torch.nn.Linear(in_channels * size * size, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10)
        )
    def forward(self, x):
        x = torch.nn.Flatten()(x)
        x = self.head(x)
        return x

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.init_block = components.InitBlock(out_channels = 16)
        self.blocks = torch.nn.ModuleList([
            components.Module(
                        conv_blocks_number = 1,
                        in_channels = 16, 
                        internal_channels = 32,
                        out_channels = 32,
                        bypass = True,
                        max_pool = False,
                        batch_norm= False,
                        dropout = False
                    ),
            components.Module(
                        conv_blocks_number = 1,
                        in_channels = 32, 
                        internal_channels = 64,
                        out_channels = 64,
                        bypass = True,
                        max_pool = True,
                        batch_norm= False,
                        dropout = False
                    ),
            components.Module(
                        conv_blocks_number = 0,
                        in_channels = 64, 
                        internal_channels = 128,
                        out_channels = 128,
                        bypass = False,
                        max_pool = False,
                        batch_norm= False,
                        dropout = False
                    )
        ]) 
        self.head = HeadBlock(in_channels = 128, size = 16)

    def forward(self, x):
        x = self.init_block(x)
        x = torch.nn.functional.relu(x)
        for it in self.blocks:
            x = it(x)
        x = self.head(x)
        return x