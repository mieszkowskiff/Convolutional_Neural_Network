from torchvision import transforms
import kornia.augmentation as K
import torch

import components

class HeadBlock(torch.nn.Module):
    def __init__(self, in_channels = 32, size = 32):
        super(HeadBlock, self).__init__()
        self.head = torch.nn.Sequential(
            torch.nn.Linear(in_channels * size * size, 2048),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(2048, 256),
            torch.nn.ReLU(inplace = True),

            torch.nn.Linear(256, 10)
        )
    def forward(self, x):
        x = torch.nn.Flatten()(x)
        x = self.head(x)
        return x

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.init_block = components.InitBlock(out_channels = 64)
        self.blocks = torch.nn.ModuleList([
            components.Module(
                        conv_blocks_number = 2,
                        in_channels = 64, 
                        internal_channels = 64,
                        out_channels = 64,
                        bypass = True,
                        max_pool = True,
                        batch_norm= True,
                        dropout = True
                    ),
            components.Module(
                        conv_blocks_number = 2,
                        in_channels = 64, 
                        internal_channels = 128,
                        out_channels = 128,
                        bypass = True,
                        max_pool = True,
                        batch_norm= True,
                        dropout = True
                    ),
            components.Module(
                        conv_blocks_number = 4,
                        in_channels = 128, 
                        internal_channels = 128,
                        out_channels = 128,
                        bypass = True,
                        max_pool = False,
                        batch_norm= True,
                        dropout = False
                    )
        ]) 
        self.head = HeadBlock(in_channels = 128, size = 8)

    def forward(self, x):
        x = self.init_block(x)
        x = torch.nn.functional.relu(x, inplace = True)
        for it in self.blocks:
            x = it(x)
        x = self.head(x)
        return x