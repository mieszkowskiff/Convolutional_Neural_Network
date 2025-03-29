from torchvision import datasets, transforms
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

sys.path.append("./read_models/init/model_init")
from model_init import initialize_model
sys.path.remove("./read_models/init/model_init")

# underdog      double_pool     new_mindfuck        new_double_pool     long_runner     no_head  
# good_no_head     damian1  uberdriver79
choose_model = "uberdriver79"

picture_index = 15000

data = "valid"

eps = 0.05

class_names = ['airplane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    model, model_path, conf_mat_name = initialize_model(choose_model)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    summary(model, (3, 32, 32))


    dataset = datasets.ImageFolder(root = f"./data/{data}", transform = transforms.ToTensor())

    picture = dataset[picture_index][0]
    label = torch.tensor([dataset[picture_index][1]]).to(device)

    plt.imshow(picture.permute(1, 2, 0))
    plt.show()

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

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


    transformed_picture = normalization(picture.unsqueeze(0)).to(device)
    
    transformed_picture.requires_grad = True
    
    criterion = torch.nn.CrossEntropyLoss()
    prediction = model(transformed_picture)
    loss = criterion(prediction, label)
    loss.backward()
    print(f"Loss without attack: {loss.item()}")
    print(f"Prediction without attack: {class_names[prediction.argmax().item()]}")

    transformed_picture = (transformed_picture + eps * torch.sign(transformed_picture.grad)).detach()

    prediction = model(transformed_picture)
    loss = criterion(prediction, label)
    print(f"Loss with attack: {loss.item()}")
    print(f"Prediction with attack: {class_names[prediction.argmax().item()]}")

    new_picture = denormalization(transformed_picture.squeeze(0)).cpu().detach()
    new_picture = torch.clamp(new_picture, 0, 1)
    plt.imshow(new_picture.permute(1, 2, 0))
    plt.show()


    





if __name__ == "__main__":
    main()