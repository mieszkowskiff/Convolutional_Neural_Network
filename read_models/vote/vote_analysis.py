from torchvision import datasets, transforms
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

torch.manual_seed(13)

sys.path.append("./read_models/init/model_init")
from model_init import initialize_model
sys.path.remove("./read_models/init/model_init")

#

choose_models = ['damian1_TUNED', "hubert1_TUNED", "hubert2_TUNED"]

acc_eval = False
acc_models = [0.805, 0.816, 0.811]

class_names = ['airplane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.47889522, 0.47227842, 0.43047404], 
            std=[0.24205776, 0.23828046, 0.25874835]
        )
    ])

    test_dataset = datasets.ImageFolder(root = "./data/valid", transform = test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1024, shuffle = False, pin_memory=True, num_workers=2)
    test_dataset_size = len(test_dataset)

    models = []
    ensembled_name = ''
    for name in choose_models:
        ensembled_name += name
        ensembled_name += '_'
    
    counter = 0
    # loading inicated models and evaluating accuracy if acc_eval = True
    for name in choose_models:
        tmp_model, model_path, _ = initialize_model(name)
        models.append(tmp_model)
        models[-1].load_state_dict(torch.load(model_path))
        models[-1].to(device)
        models[-1].eval()
        print(name + " model loaded.")
        #summary(model, (3, 32, 32))
        if(acc_eval):
            correctly_predicted = 0
            with torch.no_grad():
                print(f"Using device: {device}")
                for images, labels in tqdm.tqdm(test_loader):
                    device_images, device_labels = images.to(device), labels.long().to(device)
                    outputs = models[-1](device_images)
                    preditctions = torch.argmax(outputs, dim = 1)
                    correctly_predicted += (preditctions == device_labels).sum().item() 
            
            acc_models[counter] = correctly_predicted / test_dataset_size
            counter += 1  
            print(f"Accuracy: {correctly_predicted / test_dataset_size}")
    
    # predicting and creating the confusion matrix
    correctly_predicted = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm.tqdm(test_loader):
            device_images = images.to(device)
            device_labels = labels.long().to(device)
            outputs = acc_models[0] * torch.nn.Softmax(dim = 1)(models[0](device_images))
            for i in range(1, len(choose_models)-1):
                outputs += acc_models[i] * torch.nn.Softmax(dim = 1)(models[i](device_images))
            outputs /= len(choose_models)
            predictions = torch.argmax(outputs, dim=1)

            correctly_predicted += (predictions == device_labels).sum().item()

            # Save for confusion matrix
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(device_labels.cpu().numpy())

    print(f"Ensembled models accuracy: {correctly_predicted / test_dataset_size:.4f}")

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    plt.show()
    
if __name__ == "__main__":
    main()
    