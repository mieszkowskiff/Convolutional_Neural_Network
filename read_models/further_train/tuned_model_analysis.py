from torchvision import datasets, transforms
from torchsummary import summary

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import torch
import tqdm
import sys

sys.path.append("..\init\model_init")
from model_init import initialize_model
sys.path.remove("..\init\model_init")

# good_no_head   damian1   hubert1   hubert2   3_head   2_head   1_head   five_twelve
# good_no_head_TUNED   damian1   hubert1   hubert2   3_head   2_head   1_head
#    76.8%
choose_model = "3_head"

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

    test_dataset = datasets.ImageFolder(root = "../../data/valid", transform = test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 256, shuffle = False, pin_memory=True, num_workers=2)
    test_dataset_size = len(test_dataset)

    model, model_path, conf_mat_name = initialize_model(choose_model)
    model_path = "./fine_tuned_models/" + choose_model + "_TUNED.pth"
    print(model_path)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    summary(model, (3, 32, 32))
    
    model.eval()
    correctly_predicted = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm.tqdm(test_loader):
            device_images = images.to(device)
            device_labels = labels.long().to(device)

            outputs = model(device_images)
            predictions = torch.argmax(outputs, dim=1)

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

    # Save to path
    plt.savefig(conf_mat_name)
    plt.close()

if __name__ == "__main__":
    main()
    