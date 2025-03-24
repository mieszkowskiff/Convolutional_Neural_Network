from torchvision import datasets, transforms
from torchsummary import summary
from torch.amp import autocast, GradScaler
import torch
import time
import tqdm

class InitBlock(torch.nn.Module):
    def __init__(self, out_channels = 32):
        super(InitBlock, self).__init__()
        self.init_conv = torch.nn.Conv2d(3, out_channels, kernel_size = 3, stride = 1, padding = 1)
    
    def forward(self, x):
        if self.training:
            x = torch.nn.Sequential(
                transforms.RandomAffine(degrees = 10, translate = (0.1, 0.1), scale = (0.8, 1.2)),
            )(x)
        x = self.init_conv(x)
        return torch.nn.functional.relu(x)

class ConvolutionalBlock(torch.nn.Module):
    def __init__(self, in_channels = 32, out_channels = 32, bypass = False): 
        super(ConvolutionalBlock, self).__init__()
        self.bypass = bypass
        self.convolution = torch.nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
    
    def forward(self, x):
        y = self.convolution(x)
        if self.bypass:
            return torch.nn.functional.relu(y) + x
        return torch.nn.functional.relu(y)

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
    
class Module(torch.nn.Module):
    def __init__(self, conv_blocks_number, in_channels = 32, internal_channels = 32, out_channels = 32, bypass = False, max_pool = False):
        super(Module, self).__init__()
        self.conv_in = ConvolutionalBlock(
                    in_channels = in_channels, 
                    out_channels = internal_channels,
                    bypass = False
                )
        
        self.conv_blocks_number = conv_blocks_number
        if(self.conv_blocks_number != 0):
            self.blocks = torch.nn.Sequential(
                *[
                    ConvolutionalBlock(
                        in_channels = internal_channels, 
                        out_channels = internal_channels,
                        bypass = bypass
                    ) for _ in range(conv_blocks_number)
                ]
            )

        self.conv_out = ConvolutionalBlock(
                    in_channels = internal_channels, 
                    out_channels = out_channels,
                    bypass = bypass
                )
        
        self.max_pool = max_pool
        if(max_pool):
            self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_in(x)
        if(self.conv_blocks_number != 0):
            x = self.blocks(x) 
        x = self.conv_out(x)
        if(self.max_pool):
            x = self.pool(x)
        return x

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.init_block = InitBlock(out_channels = 16)
        self.blocks = torch.nn.ModuleList([
            Module(
                        conv_blocks_number = 1,
                        in_channels = 16, 
                        internal_channels = 32,
                        out_channels = 32,
                        bypass = True,
                        max_pool = False
                    ),
            Module(
                        conv_blocks_number = 1,
                        in_channels = 32, 
                        internal_channels = 64,
                        out_channels = 64,
                        bypass = True,
                        max_pool = True
                    ),
            Module(
                        conv_blocks_number = 0,
                        in_channels = 64, 
                        internal_channels = 128,
                        out_channels = 128,
                        bypass = False,
                        max_pool = False
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
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.47889522, 0.47227842, 0.43047404], 
            std=[0.24205776, 0.23828046, 0.25874835]
        )
    ])
    train_dataset = datasets.ImageFolder(root = "./data/train", transform = transform)


    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size = 256, 
        shuffle = True, 
        pin_memory = True, 
        num_workers = 4
        )

    test_dataset = datasets.ImageFolder(root = "./data/valid", transform = transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 512, shuffle = False, pin_memory=True, num_workers=4)
    test_dataset_size = len(test_dataset)



    model = Network()

    model.to(device)
    summary(model, (3, 32, 32))

    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    for epoch in range(30):
        print(f"Using device: {device}")
        start_time = time.time()
        model.train()
        scaler = GradScaler()
        total_loss = 0
        for images, labels in tqdm.tqdm(train_loader):
            torch.cuda.empty_cache()
            device_images, device_labels = images.to(device), labels.long().to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(device_images)
                loss = criterion(outputs, device_labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            
        end_time = time.time()
        correctly_predicted = 0
        model.eval()
        with torch.no_grad():
            print(f"Using device: {device}")
            for images, labels in tqdm.tqdm(test_loader):
                device_images, device_labels = images.to(device), labels.long().to(device)
                outputs = model(device_images)
                preditctions = torch.argmax(outputs, dim = 1)
                correctly_predicted += (preditctions == device_labels).sum().item()
        print(f"Time for testing: {time.time() - end_time}")


        print(f"Epoch {epoch + 1}, Training Loss: {total_loss}, Accuracy: {correctly_predicted / test_dataset_size}, Time: {end_time - start_time}s")
    filename = input("Enter the filename to save the model: ")
    torch.save(model.state_dict(), f"./models/{filename}.pth")
if __name__ == "__main__":
    main()