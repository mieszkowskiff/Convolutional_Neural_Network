from torchvision import datasets, transforms
import torch
import time



def main():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(root = "./data/train", transform = transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

    test_dataset = datasets.ImageFolder(root = "./data/test", transform = transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 32, shuffle = False)

    model = torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels = 3, 
            out_channels = 16, 
            kernel_size = 3, 
            stride = 1, 
            padding = 1
        ),
        torch.nn.ReLU(),
        torch.nn.Conv2d(
            in_channels = 16, 
            out_channels = 32, 
            kernel_size = 3, 
            stride = 1, 
            padding = 1
        ),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size = 2),
        torch.nn.Flatten(),
        torch.nn.Linear(32 * 16 * 16, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10)
        )
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    for epoch in range(10):
        start_time = time.time()
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss}, Time: {time.time() - start_time}")

if __name__ == "__main__":
    main()