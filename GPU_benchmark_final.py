import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
import time

# Model Head
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

# Full Network
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

    torch.backends.cudnn.benchmark = True

    # Load and preload MNIST to RAM
    transform = transforms.ToTensor()
    train_raw = MNIST(root="./data", train=True, download=True, transform=transform)
    test_raw = MNIST(root="./data", train=False, download=True, transform=transform)

    train_images = torch.stack([train_raw[i][0] for i in range(len(train_raw))])  # [60000, 1, 28, 28]
    train_labels = torch.tensor([train_raw[i][1] for i in range(len(train_raw))], dtype=torch.long)

    test_images = torch.stack([test_raw[i][0] for i in range(len(test_raw))])
    test_labels = torch.tensor([test_raw[i][1] for i in range(len(test_raw))], dtype=torch.long)

    model = Network().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    batch_size = 512

    # Training Loop
    for epoch in range(5):
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
