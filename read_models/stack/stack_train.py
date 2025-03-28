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
choose_models = ['long_runner', 'new_double_pool', 'underdog', 'double_pool']

class MetaStackingHead(nn.Module):
    def __init__(self, input_dim=40, hidden_dims=[128, 64], num_classes=10, dropout=0.3):
        super(MetaStackingHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            #nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            #nn.Dropout(dropout),

            nn.Linear(hidden_dims[0], hidden_dims[1]),
            #nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(inplace=True),
            #nn.Dropout(dropout),

            nn.Linear(hidden_dims[1], num_classes)
        )

    def forward(self, x):
        return self.net(x)

class StackedEnsemble(nn.Module):
    def __init__(self, models, meta_head):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.meta_head = meta_head

    def forward(self, x):
        logits = [model(x) for model in self.models]
        x = torch.cat(logits, dim=1)
        return self.meta_head(x)

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

    models = []

    for name in choose_models:
        tmp_model, model_path, _ = initialize_model(name)
        models.append(tmp_model)
        models[-1].load_state_dict(torch.load(model_path))
        models[-1].eval()
        print(name + " model loaded.")
        #summary(model, (3, 32, 32))
    
    meta_head = MetaStackingHead()
    ensemble = StackedEnsemble(models = models, meta_head = meta_head)
    ensemble.to(device)
    # freeze conv models parameters, only meta head is being trained
    for model in ensemble.models:
        for param in model.parameters():
            param.requires_grad = False

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ensemble.meta_head.parameters(), lr = 0.01, weight_decay=1e-4)
    # monitor best acc model and save it to chechpoint
    best_acc = 0

    for epoch in range(2):
        print(f"Using device: {device}")
        start_time = time.time()
        meta_head.train()
        scaler = GradScaler()
        total_loss = 0
        for images, labels in tqdm.tqdm(train_loader):
            torch.cuda.empty_cache()
            device_images, device_labels = images.to(device), labels.long().to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                outputs = ensemble(device_images)
                loss = criterion(outputs, device_labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            
        end_time = time.time()
        correctly_predicted = 0
        meta_head.eval()
        with torch.no_grad():
            print(f"Using device: {device}")
            for images, labels in tqdm.tqdm(test_loader):
                device_images, device_labels = images.to(device), labels.long().to(device)
                outputs = ensemble(device_images)
                preditctions = torch.argmax(outputs, dim = 1)
                correctly_predicted += (preditctions == device_labels).sum().item()
        print(f"Time for testing: {time.time() - end_time}")
        acc = correctly_predicted / test_dataset_size
        print(f"Epoch {epoch + 1}, Training Loss: {total_loss}, Accuracy: {acc}, Time: {end_time - start_time}s")
        if(acc>best_acc):
            torch.save(copy.deepcopy(ensemble.meta_head.state_dict()), f"./heads/checkpoint/model.pth")
            torch.save(optimizer.state_dict(), f"./heads/checkpoint/optimizer.pth")
            best_acc = acc

    # at the end of the training, type the name of the model, it will move the best model instance 
    # from chechpoint to models directory and name the model and optimizer files accordingly to the name 
    filename = input("Enter the model name to save the model and optimizer: ")
    model_name = filename + "_model"
    opt_name = filename + "_optim"
    shutil.move("./heads/checkpoint/model.pth", f"./heads/{model_name}.pth")
    shutil.move("./heads/checkpoint/optimizer.pth", f"./heads/{opt_name}.pth")

if __name__ == "__main__":
    main()
    