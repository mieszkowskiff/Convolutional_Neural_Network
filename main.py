from torchvision import datasets, transforms
import torch
import time

class Block(torch.nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.convolutions = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 16, kernel_size = 3, stride = 1, padding = 1)
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(32, 16, kernel_size = 3, stride = 1, padding = 1)
            )
        ])
    
    def forward(self, x):
        y = torch.cat([conv(x) for conv in self.convolutions], dim = 1)
        y = torch.nn.functional.relu(x)
        return y + x

class Network(torch.nn.Module):
    def __init__(self, blocks_number):
        super(Network, self).__init__()
        self.init_conv = torch.nn.Conv2d(3, 32, kernel_size = 3, stride = 1, padding = 1)
        self.blocks = torch.nn.Sequential(*[Block() for _ in range(blocks_number)])
        self.linear = torch.nn.Linear(32 * 32 * 32, 10)

    def forward(self, x):
        x = self.init_conv(x)
        x = torch.nn.functional.relu(x)
        x = self.blocks(x)
        x = torch.nn.Flatten()(x)
        x = self.linear(x)
        return x
    



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(root = "./data/train", transform = transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True, pin_memory=True, num_workers=4)

    test_dataset = datasets.ImageFolder(root = "./data/test", transform = transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 128, shuffle = False, pin_memory=True, num_workers=4)
    test_dataset_size = len(test_dataset)



    model = Network(5)

    model.to(device)

    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    for epoch in range(10):
        start_time = time.time()
        total_loss = 0
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.long().to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        end_time = time.time()
        correctly_predicted = 0
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.long().to(device)
                outputs = model(images)
                preditctions = torch.argmax(outputs, dim = 1)
                correctly_predicted += (preditctions == labels).sum().item()
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        print(f"Time for testing: {time.time() - end_time}")


        print(f"Epoch {epoch + 1}, Loss: {total_loss}, Accuracy: {correctly_predicted / test_dataset_size}, Time: {end_time - start_time}s")

if __name__ == "__main__":
    main()