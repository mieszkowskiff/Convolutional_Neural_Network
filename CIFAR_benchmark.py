import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import time
from tqdm import tqdm

def compute_mean_std(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=2)

    mean = 0.0
    std = 0.0
    total_images = 0

    for images, _ in loader:
        batch_samples = images.size(0)  # batch size (might be smaller at end)
        images = images.view(batch_samples, 3, -1)  # flatten H and W
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    return mean, std

def preload_dataset(dataset, batch_size=512, num_workers=4, device='cpu'):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    all_images = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Preloading dataset"):
        all_images.append(images)
        all_labels.append(labels)

    images_tensor = torch.cat(all_images)
    labels_tensor = torch.cat(all_labels)

    return images_tensor.to(device), labels_tensor.to(device)

class Conv_Block(torch.nn.Module):
    def __init__(self, in_out_channels = 8, internal_channels = 8, kernel_size = 3, stride = 1, padding = 1): 
        super(Conv_Block, self).__init__()
        self.convolutions = torch.nn.Sequential(
            torch.nn.Conv2d(in_out_channels, internal_channels, kernel_size, stride, padding),
            torch.nn.ReLU()
        )
    
    def forward(self, x):
        return self.convolutions(x)
    
class Res_Conv_Block(torch.nn.Module):
    def __init__(self, in_out_channels = 8, internal_channels = 8, kernel_size = 3, stride = 1, padding = 1): 
        super(Res_Conv_Block, self).__init__()
        self.convolutions = torch.nn.Sequential(
            torch.nn.Conv2d(in_out_channels, internal_channels, kernel_size, stride, padding)
        )
    
    def forward(self, x):
        y = self.convolutions(x)
        x = torch.nn.functional.relu(x + y)
        return x

class Pool_Block(torch.nn.Module):
    def __init__(self, kernel_size = 2, stride = 2): 
        super(Pool_Block, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride)
    
    def forward(self, x):
        return self.pool(x)

# Model Head
class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10)   
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.head(x)

# Full Network
class Network(nn.Module):
    def __init__(self, in_out_channels=32, internal_channels=64):
        super().__init__()
        self.init_conv = nn.Conv2d(3, internal_channels, kernel_size=3, padding=1)
        self.blocks1 = torch.nn.Sequential(
            *[
                Res_Conv_Block(in_out_channels=64, internal_channels=64) for _ in range(5)
            ],
            Pool_Block(kernel_size = 2, stride = 2)
        )
        self.blocks2 = torch.nn.Sequential(
            Conv_Block(in_out_channels=64, internal_channels=128),
            *[
                Conv_Block(in_out_channels=128, internal_channels=128) for _ in range(6)
            ],
            Pool_Block(kernel_size = 2, stride = 2)
        )
        self.head = Head()

    def forward(self, x):
        x = F.relu(self.init_conv(x))
        x = self.blocks1(x)
        x = self.blocks2(x)
        return self.head(x)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.backends.cudnn.benchmark = True

    # Load and preload MNIST to RAM  
    print("Loading the dataset...")
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    print("Evaluating the parameters for normalization...")
    mean_std_path = "mean_std.pt"
    if os.path.exists(mean_std_path):
        print("Loading cached mean and std...")
        mean, std = torch.load(mean_std_path)
    else:
        print("Computing mean and std from dataset...")
        mean, std = compute_mean_std(train_dataset)
        torch.save((mean, std), mean_std_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    print("Data and label separation...")

    train_dataset = CIFAR10(root='./data', train=True, download=False, transform=transform)

    if os.path.exists("train_images.pt") and os.path.exists("train_labels.pt"):
        print("Loading cached training data...")
        train_images = torch.load("train_images.pt")
        train_labels = torch.load("train_labels.pt")
    else:
        print("Preloading training data for the first time...")
        train_images, train_labels = preload_dataset(train_dataset, batch_size=512, num_workers=4, device='cpu')
        torch.save(train_images, "train_images.pt")
        torch.save(train_labels, "train_labels.pt")

    if os.path.exists("test_images.pt") and os.path.exists("test_labels.pt"):
        print("Loading cached test data...")
        test_images = torch.load("test_images.pt")
        test_labels = torch.load("test_labels.pt")
    else:
        print("Preloading test data for the first time...")
        test_images, test_labels = preload_dataset(test_dataset, batch_size=512, num_workers=4, device='cpu')
        torch.save(test_images, "test_images.pt")
        torch.save(test_labels, "test_labels.pt")

    model = Network().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    batch_size = 128
    print(f'Model created and moved to {device}.')

    print("Training Loop started...")
    # Training Loop
    for epoch in range(30):
        model.train()
        start_time = time.time()
        total_loss = 0

        for i in range(0, len(train_images), batch_size):
            images = train_images[i:i+batch_size].to(device)
            labels = train_labels[i:i+batch_size].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        end_time = time.time()
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f} | Time = {end_time - start_time:.2f}s")

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(test_images), batch_size):
                images = test_images[i:i+batch_size].to(device)
                labels = test_labels[i:i+batch_size].to(device)

                outputs = model(images)
                predicted = outputs.argmax(dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        acc = 100 * correct / total
        print(f"Test Accuracy after epoch {epoch+1}: {acc:.2f}%")

if __name__ == "__main__":
    main()
