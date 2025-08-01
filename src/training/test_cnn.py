from utils import *
import torch.optim as optim
from sklearn.model_selection import KFold
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset, Dataset
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import Counter
from torchmetrics import Accuracy

def plot_confusion_matrix(TP, TN, FP, FN, dir):

    cm = np.array([[TP, FP],
                   [FN, TN]])

    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(cm, cmap='Blues')
    plt.colorbar(cax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(['Predicted Positive', 'Predicted Negative'], rotation=45)
    ax.set_yticklabels(['Actual Positive', 'Actual Negative'])

    # Annotate cells with counts
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f"{val}", ha='center', va='center', fontsize=14, color='black')

    plt.title(f'Confusion Matrix of Model on Test Set')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    plt.savefig(dir / f"test_confusion.png", dpi=300)
    plt.close()


def save_test_data(test_accuracy, precision, recall, f1_score, dir):
    with open(dir / 'test_results.txt', 'w') as f:
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Precision:     {precision:.4f}\n")
        f.write(f"Recall:        {recall:.4f}\n")
        f.write(f"F1 Score:      {f1_score:.4f}\n")
        f.close()

class PTBatchDataset(Dataset):
    def __init__(self, pt_file_paths, dir):
        self.pt_file_paths = [Path(p) for p in pt_file_paths]
        self.data_dir = dir

        # Build a global index map: list of (file_idx, local_index)
        self.index_map = []
        self.file_sizes = []
        for file_idx, file_path in enumerate(self.pt_file_paths):
            print(f"Loading {self.data_dir / file_path}")
            data = torch.load(self.data_dir / file_path)
            print(f"Loaded {self.data_dir / file_path}")
            size = len(data['target'])
            self.file_sizes.append(size)
            self.index_map.extend([(file_idx, i) for i in range(size)])

        # Cache for currently loaded file
        self.current_file_idx = None
        self.cached_data = None

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, local_idx = self.index_map[idx]
        if self.current_file_idx != file_idx:
            # Cache new file
            path = self.data_dir / self.pt_file_paths[file_idx]

            self.cached_data = torch.load(path)
            self.current_file_idx = file_idx

        image = self.cached_data['image'][local_idx]
        target = self.cached_data['target'][local_idx]

        return image, target

if __name__ == '__main__':

    script_dir = Path(__file__).parent.resolve()
    test_dir = script_dir / 'test_data'
    figures_dir = script_dir / 'figures'
    model_dir = script_dir / 'model'

    figures_dir.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    torch.cuda.empty_cache()

    # load files into the dataset
    files = get_filenames(test_dir, ext='.pt')
    test_dataset = PTBatchDataset(files, test_dir)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier[0] = nn.Dropout(p=0.3)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    checkpoint = torch.load(model_dir / 'model_fold_1.pth', map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    with torch.no_grad():

        total_samples = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            batch_size = inputs.size(0)

            outputs = model(inputs)

            total_samples += batch_size

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            labels = labels.long().view(-1)

            TP += ((preds == 1) & (labels == 1)).sum().item()
            TN += ((preds == 0) & (labels == 0)).sum().item()
            FP += ((preds == 1) & (labels == 0)).sum().item()
            FN += ((preds == 0) & (labels == 1)).sum().item()

    test_accuracy = (TP + TN) / total_samples
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    plot_confusion_matrix(TP, TN, FP, FN, figures_dir)
    save_test_data(test_accuracy, precision, recall, f1_score, figures_dir)