from torchvision import datasets, transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torchsummary import summary
from torch.amp import autocast, GradScaler
import torch.nn as nn
import kornia.augmentation as K

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import torch
import time
import tqdm
import sys
import copy
import shutil

sys.path.append("..\init\model_init")
from model_init import initialize_model
sys.path.remove("..\init\model_init")

#'underdog' - 75.5  'double_pool' - 74.8  'new_mindfuck' - 70  'new_double_pool' - 75.5  'long_runner' - 72.1          
#choose_models = ['long_runner', 'new_double_pool', 'underdog', 'new_mindfuck', 'double_pool']
choose_model = "underdog"

class_names = ['airplane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # again, be cautions with transforms, separate for train and test since AutoAugment
    train_transform = transforms.Compose([
        AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.47889522, 0.47227842, 0.43047404], 
            std=[0.24205776, 0.23828046, 0.25874835]
        )
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.47889522, 0.47227842, 0.43047404], 
            std=[0.24205776, 0.23828046, 0.25874835]
        )
    ])

    train_dataset = datasets.ImageFolder(root = "../../data/train", transform = train_transform)
    test_dataset = datasets.ImageFolder(root = "../../data/valid", transform = test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size = 512, 
        shuffle = True, 
        pin_memory = True, 
        num_workers = 2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size = 512, 
        shuffle = False, 
        pin_memory=True, 
        num_workers=2
    )
    test_dataset_size = len(test_dataset)

    model, model_path, conf_mat_path = initialize_model(choose_model)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    summary(model, (3, 32, 32))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    opt_path = '../../models/' + choose_model + '_optim.pth'
    optimizer.load_state_dict(opt_path)

    for epoch in range(1):
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
        all_preds = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            print(f"Using device: {device}")
            for images, labels in tqdm.tqdm(test_loader):
                device_images, device_labels = images.to(device), labels.long().to(device)
                outputs = model(device_images)
                predictions = torch.argmax(outputs, dim = 1)
                correctly_predicted += (predictions == device_labels).sum().item()

                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(device_labels.cpu().numpy())
        print(f"Time for testing: {time.time() - end_time}")
        acc = correctly_predicted / test_dataset_size
        print(f"Epoch {epoch + 1}, Training Loss: {total_loss}, Accuracy: {acc}, Time: {end_time - start_time}s")
        if(acc>best_acc):
            torch.save(copy.deepcopy(model.state_dict()), f"./models/checkpoint/model.pth")
            torch.save(optimizer.state_dict(), f"./models/checkpoint/optimizer.pth")
            best_acc = acc
    
    filename = input("Enter the filename to save the model: ")

    torch.save(copy.deepcopy(model.state_dict()), f"./fine_tuned_models/{filename}.pth")
    torch.save(model.state_dict(), f"./fine_tuned_models/{filename}.pth")

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # Save to path
    plt.savefig(conf_mat_path)
    plt.close()

    
    
    # at the end of the training, type the name of the model, it will move the best model instance 
    # from chechpoint to models directory and name the model and optimizer files accordingly to the name 
    filename = input("Enter the model name to save the model and optimizer: ")
    model_name = filename
    opt_name = filename + "_optim"
    shutil.move("./models/checkpoint/model.pth", f"./models/{model_name}.pth")
    shutil.move("./models/checkpoint/optimizer.pth", f"./models/{opt_name}.pth")

if __name__ == "__main__":
    main()
    