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

from stack_train import StackedEnsemble

sys.path.append("./read_models/init/model_init")
from model_init import initialize_model
sys.path.remove("./read_models/init/model_init")

sys.path.append("./read_models/stack/head_init")
from head_init import initialize_head
sys.path.remove("./read_models/stack/head_init")

#good_no_head   damian1   3_head   2_head   1_head   hubert1   hubert2

#choose_models = ['good_no_head', 'damian1', 'hubert1', 'hubert2', '1_head', '2_head', '3_head']
choose_models = ["uberdriver79", 'damian1_TUNED', 'hubert1_TUNED', 'hubert2_TUNED']
choose_head = "small_head"

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
        print(model_path)
        models.append(tmp_model)
        models[-1].load_state_dict(torch.load(model_path))
        models[-1].eval()
        print(name + " model loaded.")

    head_path = "./read_models/stack/heads/" + choose_head + ".pth"
    meta_head = initialize_head(choose_head)
    meta_head.load_state_dict(torch.load(head_path))
    ensemble = StackedEnsemble(models = models, meta_head = meta_head)
    ensemble.to(device)
    
    summary(ensemble, (3, 32, 32))

    for param in ensemble.parameters():
        param.requires_grad = False

    ensemble.eval()
    correctly_predicted = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        print(f"Using device: {device}")
        for images, labels in tqdm.tqdm(test_loader):
            device_images, device_labels = images.to(device), labels.long().to(device)

            outputs = ensemble(device_images)
            predictions = torch.argmax(outputs, dim = 1)
            
            correctly_predicted += (predictions == device_labels).sum().item()
            
            # Save for confusion matrix
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(device_labels.cpu().numpy())
    
    print(f"Accuracy: {correctly_predicted / test_dataset_size:.4f}")

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    plt.show()
    plt.close()

if __name__ == "__main__":
    main()