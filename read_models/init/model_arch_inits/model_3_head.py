from torchvision import transforms
import kornia.augmentation as K
import torch

import components

class HeadBlock(torch.nn.Module):
    def __init__(self, in_channels = 32, size = 32):
        super(HeadBlock, self).__init__()
        self.head = torch.nn.Sequential(
            torch.nn.Linear(in_channels * size * size, 1024),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(1024, 256),
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
        self.init_block = components.InitBlock(out_channels = 128)
        self.blocks = torch.nn.ModuleList([
            components.Module(
                        conv_blocks_number = 0,
                        in_channels = 128, 
                        internal_channels = 128,
                        out_channels = 128,
                        bypass = True,
                        max_pool = True,
                        batch_norm= True,
                        dropout = False
                    ),
            components.Module(
                        conv_blocks_number = 2,
                        in_channels = 128, 
                        internal_channels = 256,
                        out_channels = 256,
                        bypass = True,
                        max_pool = False,
                        batch_norm= True,
                        dropout = False
                    ),
            components.Module(
                        conv_blocks_number = 4,
                        in_channels = 256, 
                        internal_channels = 512,
                        out_channels = 128,
                        bypass = True,
                        max_pool = False,
                        batch_norm= True,
                        dropout = False
                    )
        ]) 

        self.head = HeadBlock(128, 16)
        
        #self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))
        #self.classifier = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = self.init_block(x)
        x = torch.nn.functional.relu(x)
        for it in self.blocks:
            x = it(x)
        #x = self.gap(x)              # [B, 256, 1, 1]
        #x = torch.flatten(x, 1)      # [B, 256]
        #x = self.classifier(x)       # [B, 10]
        x = self.head(x)
 
        return x