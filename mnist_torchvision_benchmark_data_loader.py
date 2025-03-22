import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import time

# CNN Head
class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(8 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.head(x)

# Full CNN
class Network(nn.Module):
    def __init__(self, in_out_channels=8, internal_channels=8):
        super().__init__()
        self.init_conv = nn.Conv2d(1, internal_channels, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(internal_channels, in_out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.head = Head()

    def forward(self, x):
        x = F.relu(self.init_conv(x))
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = F.relu(self.conv(x))
        return self.head(x)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Best practice: torch.backends.cudnn.benchmark for fixed input shapes
    torch.backends.cudnn.benchmark = True

    # Load datasets
    transform = transforms.ToTensor()
    train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = MNIST(root="./data", train=False, download=True, transform=transform)

    # DataLoaders: batched loading from CPU with pin_memory + workers
    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = Network().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(5):
        model.train()
        start_time = time.time()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)

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
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, dtype=torch.long, non_blocking=True)

                outputs = model(images)
                predicted = outputs.argmax(dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        acc = 100 * correct / total
        print(f"Test Accuracy after epoch {epoch+1}: {acc:.2f}%")

if __name__ == "__main__":
    main()
