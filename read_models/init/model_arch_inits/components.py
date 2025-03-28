from torchvision import transforms
import kornia.augmentation as K
import torch

class InitBlock(torch.nn.Module):
    def __init__(self, out_channels = 32):
        super(InitBlock, self).__init__()
        self.init_conv = torch.nn.Conv2d(3, out_channels, kernel_size = 3, stride = 1, padding = 1)
    
    def forward(self, x):
        # using AutoAugment no custom Augmentation
        # we are doing the augmentation via transforms.Compose()
        if False:
            x = torch.nn.Sequential(
                transforms.RandomAffine(degrees = 14, translate = (0.1, 0.1), scale = (0.7, 1.3)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomVerticalFlip(p=0.3),
                K.RandomGaussianBlur((3, 3), (0.1, 0.3), p=0.2),
                K.RandomGaussianNoise(mean=0.0, std=0.02, p=0.2)
            )(x)
        x = self.init_conv(x)
        return torch.nn.functional.relu(x, inplace = True)

class ConvolutionalBlock(torch.nn.Module):
    def __init__(self, in_channels = 32, out_channels = 32, bypass = False, batch_norm = False): 
        super(ConvolutionalBlock, self).__init__()
        self.bypass = bypass
        self.batch_norm = batch_norm
        if(self.batch_norm):
            self.block = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                torch.nn.BatchNorm2d(out_channels) 
            )        
        else:
            self.block = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
            )

    def forward(self, x):
        y = self.block(x)
        if self.bypass:
            return torch.nn.functional.relu(x + y, inplace = True)
        return torch.nn.functional.relu(y, inplace = True)
    
class Module(torch.nn.Module):
    def __init__(self, 
                conv_blocks_number, 
                in_channels = 32, 
                internal_channels = 32, 
                out_channels = 32, 
                bypass = False, 
                max_pool = False,
                batch_norm = False,
                dropout = False
                ):
        super(Module, self).__init__()
        self.conv_in = ConvolutionalBlock(
                    in_channels = in_channels, 
                    out_channels = internal_channels,
                    bypass = False,
                    batch_norm = batch_norm
                )
        
        self.conv_blocks_number = conv_blocks_number
        if(self.conv_blocks_number != 0):
            self.blocks = torch.nn.Sequential(
                *[
                    ConvolutionalBlock(
                        in_channels = internal_channels, 
                        out_channels = internal_channels,
                        bypass = bypass,
                        batch_norm = batch_norm
                    ) for _ in range(conv_blocks_number)
                ]
            )

        self.conv_out = ConvolutionalBlock(
                    in_channels = internal_channels, 
                    out_channels = out_channels,
                    bypass = False,
                    batch_norm = batch_norm
                )
        
        self.max_pool = max_pool
        if(max_pool):
            self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = dropout
        if(dropout):
            self.dropout_layer = torch.nn.Dropout2d(0.1)

    def forward(self, x):
        x = self.conv_in(x)
        if(self.conv_blocks_number != 0):
            x = self.blocks(x) 
        x = self.conv_out(x)
        if(self.max_pool):
            x = self.pool(x)
        if(self.dropout):
            x = self.dropout_layer(x)
        return x