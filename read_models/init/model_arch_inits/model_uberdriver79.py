from torchvision import transforms
import torch

import components



class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.init_block = components.InitBlock(out_channels = 256)
        self.blocks = torch.nn.ModuleList([
            components.Module(
                        conv_blocks_number = 12,
                        in_channels = 256, 
                        internal_channels = 256,
                        out_channels = 256,
                        bypass = True,
                        max_pool = False,
                        batch_norm = True,
                        dropout = False
                    )
        ]) 
        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = self.init_block(x)
        x = torch.nn.functional.relu(x)
        for it in self.blocks:
            x = it(x)
        x = self.gap(x)              # [B, 256, 1, 1]
        x = torch.flatten(x, 1)      # [B, 256]
        x = self.classifier(x)       # [B, 10]
        return x