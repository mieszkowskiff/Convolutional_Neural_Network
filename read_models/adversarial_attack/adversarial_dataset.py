from torchvision import datasets, transforms, utils
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torchsummary import summary
from torch.amp import autocast, GradScaler
import kornia.augmentation as K

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import torch
import time
import tqdm
import sys
import os

import shutil

sys.path.append("./read_models/init/model_init")
from model_init import initialize_model
sys.path.remove("./read_models/init/model_init")

# underdog      double_pool     new_mindfuck        new_double_pool     long_runner     no_head  
# good_no_head     damian1  uberdriver79
choose_model = "uberdriver79"

data = "train"

new_data_path = "./data/adversarial_data001"


eps = 0.02

class_names = ['airplane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def main():
    shutil.rmtree(new_data_path, ignore_errors=True)  
    os.makedirs(new_data_path, exist_ok=True)
    for class_name in class_names:
        os.makedirs(f"{new_data_path}/{class_name}", exist_ok=True) 



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    model, model_path, conf_mat_name = initialize_model(choose_model)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    summary(model, (3, 32, 32))

    mean = [0.47889522, 0.47227842, 0.43047404]
    std = [0.24205776, 0.23828046, 0.25874835]
    normalization = transforms.Normalize(
        mean=mean, 
        std=std
    )

    denormalization = transforms.Normalize(
    mean=[-m / s for m, s in zip(mean, std)], 
    std=[1 / s for s in std]
    )

    dataset = datasets.ImageFolder(
        root = f"./data/{data}", 
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalization
        ])
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size = 1, 
        shuffle = False, 
        pin_memory = True, 
        num_workers = 2
    )

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    criterion = torch.nn.CrossEntropyLoss()

    for i, (picture, label) in enumerate(tqdm.tqdm(dataloader)):
        picture = picture.to(device)
        label = label.to(device)

        transformed_picture = picture.clone()
        transformed_picture.requires_grad = True

        prediction = model(transformed_picture)
        loss = criterion(prediction, label)
        loss.backward()

        transformed_picture = (transformed_picture + eps * torch.sign(transformed_picture.grad)).detach()

        text_label = class_names[label.item()]

        picture_back = denormalization(picture.squeeze(0)).cpu().detach()
        picture_back = torch.clamp(picture_back, 0, 1)
        utils.save_image(picture_back, f"{new_data_path}/{text_label}/{i}.png")



        




    



if __name__ == "__main__":
    main()
    