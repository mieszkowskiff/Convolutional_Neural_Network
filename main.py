from torchvision import datasets, transforms
import torch
import time

class Block(torch.nn.Module):
    def __init__(self, in_out_channels = 32, internal_channels = 32): 
        super(Block, self).__init__()
        self.convolutions = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(in_out_channels, internal_channels, kernel_size = 3, stride = 1, padding = 1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(internal_channels, in_out_channels, kernel_size = 3, stride = 1, padding = 1),
            )
        ])
    
    def forward(self, x, bypass = True):
        y = torch.cat([conv(x) for conv in self.convolutions], dim = 1)
        if bypass:
            x = torch.nn.functional.relu(x + y)
        else:
            x = torch.nn.functional.relu(y)
        return x

class Head(torch.nn.Module):
    def __init__(self, in_out_channels = 32):
        super(Head, self).__init__()
        self.head = torch.nn.Sequential(
            torch.nn.Linear(in_out_channels * 32 * 32, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = torch.nn.Flatten()(x)
        x = self.head(x)
        return x


class Network(torch.nn.Module):
    def __init__(self, blocks_number, in_out_channels = 32, internal_channels = 32):
        super(Network, self).__init__()
        self.init_conv = torch.nn.Conv2d(3, in_out_channels, kernel_size = 3, stride = 1, padding = 1)
        self.blocks = torch.nn.Sequential(
            *[
                Block(
                    in_out_channels = in_out_channels, 
                    internal_channels = internal_channels
                ) for _ in range(blocks_number)
            ]
        )
        self.head = Head(in_out_channels = in_out_channels)

    def forward(self, x):
        x = self.init_conv(x)
        x = torch.nn.functional.relu(x)
        x = self.blocks(x)
        x = self.head(x)
        return x
    



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_transform = transforms.ToTensor()

    train_dataset_usage = 0.1

    train_dataset = datasets.ImageFolder(root = "./data/train", transform = train_transform)


    dataset_size = len(train_dataset)
    subset_size = int(dataset_size * train_dataset_usage)
    random_indices = torch.randperm(dataset_size)[:subset_size]
    train_subset = torch.utils.data.Subset(train_dataset, random_indices)

    train_loader = torch.utils.data.DataLoader(
        train_subset, 
        batch_size = 32, 
        shuffle = True, 
        pin_memory = True, 
        num_workers = 4
        )

    test_dataset = datasets.ImageFolder(root = "./data/test", transform = test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 128, shuffle = False, pin_memory=True, num_workers=4)
    test_dataset_size = len(test_dataset)



    model = Network(
        blocks_number = 5,
        in_out_channels = 128,
        internal_channels = 64
        )

    model.to(device)

    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    for epoch in range(5):
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
    filename = input("Enter the filename to save the model: ")
    torch.save(model.state_dict(), f"./models/{filename}.pth")
if __name__ == "__main__":
    main()