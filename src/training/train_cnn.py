from cnn_models import *
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

# file path constants
script_dir = Path(__file__).parent.resolve()
models_dir = script_dir / 'models'
figures_dir = script_dir / 'figures'
training_dir = script_dir / 'training_data'
models_dir.mkdir(exist_ok=True)
figures_dir.mkdir(exist_ok=True)

def plot_loss(train, val, fold):

    train = [t.cpu().item() if torch.is_tensor(t) else t for t in train]
    val = [v.cpu().item() if torch.is_tensor(v) else v for v in val]

    epochs = range(1, len(train) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train, label='Training Loss', marker='o')
    plt.plot(epochs, val, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(figures_dir / f"model_fold_{fold+1}_loss.png", dpi=300)
    plt.close()

def plot_accuracy(train, val, fold):

    train = [t.cpu().item() if torch.is_tensor(t) else t for t in train]
    val = [v.cpu().item() if torch.is_tensor(v) else v for v in val]

    epochs = range(1, len(train) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train, label='Training Accuracy', marker='o')
    plt.plot(epochs, val, label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(figures_dir / f"model_fold_{fold+1}_accuracy.png", dpi=300)
    plt.close()

def plot_confusion_matrix(confusion_arr, fold):

    TP, TN, FP, FN = [
        int(x.cpu()) if torch.is_tensor(x) else int(x)
        for x in confusion_arr[-1]
    ]

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

    plt.title(f'Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    plt.savefig(figures_dir / f"model_fold_{fold+1}_confusion.png", dpi=300)
    plt.close()

def plot_prf(confusion_arr, fold):

    cleaned = []
    for row in confusion_arr:
        if torch.is_tensor(row[0]):
            cleaned.append([x.detach().cpu().item() for x in row])
        else:
            cleaned.append(row)

    TP, TN, FP, FN = zip(*cleaned)
    epochs = range(1, len(confusion_arr) + 1)

    precision = [tp / (tp + fp) if (tp + fp) > 0 else 0 for tp, fp in zip(TP, FP)]
    recall = [tp / (tp + fn) if (tp + fn) > 0 else 0 for tp, fn in zip(TP, FN)]
    f1_score = [
        (2 * p * r) / (p + r) if (p + r) > 0 else 0
        for p, r in zip(precision, recall)
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, precision, label='Precision', marker='o')
    plt.plot(epochs, recall, label='Recall', marker='o')
    plt.plot(epochs, f1_score, label='F1 Score', marker='o')

    plt.title('Precision, Recall, and F1 Score Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(figures_dir / f"model_fold_{fold+1}_prf.png", dpi=300)
    plt.close()

def get_class_weights(npz_file_paths):
    label_counts = Counter()
    for path in npz_file_paths:
        with np.load(training_dir / path) as data:
            for label in data['target']:
                label_counts[int(label)] += 1
    
    if not label_counts:
        raise ValueError("No labels found in the dataset.")

    if len(label_counts.keys()) <= 1:
        raise ValueError(f"Only singular label in dataset: {label_counts}")
    
    print(label_counts)

    # note: the number of classes is based on the largest integer label
    total = sum(label_counts.values())
    classes = max(label_counts.keys())

    class_weights = []

    for i in range(classes+1):
        count = label_counts.get(i, 0)
        print(f"Number of class {i}: {count}")
        if count > 0:
            weight = total / (count * classes)
        else:
            raise ValueError(f"No labels found in dataset with value {i} despite max value being {classes}: {label_counts}")
        class_weights.append(weight)
    
    return class_weights

    

# custom dataset to ensure not all images need to be loaded into memory at once
class NPZBatchDataset(Dataset):
    def __init__(self, npz_file_paths, transform=None):
        self.transform = transform
        all_images = []
        all_labels = []

        # Load everything into memory at once
        for path in npz_file_paths:
            data = np.load(training_dir / path)
            images = data['image']   # shape: (N, C, H, W) or (N, H, W), depending on your data
            labels = data['target'] # shape: (N,)
            all_images.append(images)
            all_labels.append(labels)

        # Stack all loaded arrays into single tensors
        self.images = torch.tensor(np.concatenate(all_images, axis=0), dtype=torch.float32, device=device)
        self.labels = torch.tensor(np.concatenate(all_labels, axis=0), dtype=torch.long, device=device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# training hyperparameters
k: int = 5
kfold = KFold(n_splits=k, shuffle=True, random_state=71)
batch_size: int = 32
epochs: int = 100
patience: int = 5

if __name__ == '__main__':

    # try to use CUDA (graphics card)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    print(f"Using {device}")
    print(f"{torch.cuda.memory_allocated()}")

    # load files into the dataset
    files = get_filenames(training_dir, ext='.npz')
    train_dataset = NPZBatchDataset(files)

    # compute the class weights for loss function (for class imbalance)
    class_weights = get_class_weights(files)
    print(f"Class weights: {class_weights}")
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float, device=device)

    classes = len(class_weights)

    # training loop
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):

        training_losses = []
        validation_losses = []
        training_accuracies = []
        validation_accuracies = []
        validation_cms = []

        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size)

        total_train_batches = len(train_loader)
        total_val_batches = len(val_loader)

        model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)

        # freeze other layers
        for param in model.parameters():
            param.requires_grad = False

        model.classifier[0] = nn.Dropout(p=0.3)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, classes)
        model.to(device=device)

        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), weight_decay=1e-5)

        criterion.to(device=device)

        total_val_samples: int = 0
        total_train_samples: int = 0

        # early stopping variables
        best_val_loss: float = float('inf')
        counter: int = 0

        for epoch in range(epochs):
            print(f"Fold [{fold+1}/{k}] - Epoch [{epoch+1}/{epochs}]")
            running_loss: float = 0.0
            train_loss: float = 0.0
            val_loss: float = 0.0
            train_accuracy: float = 0.0
            total_train_samples: int = 0
            val_accuracy: float = 0.0
            total_val_samples: int = 0

            # confusion matrix
            TP: int = 0
            TN: int = 0
            FP: int = 0
            FN: int = 0
            correct: int = 0

            model.train()
            for i, data in enumerate(train_loader):
                print(f"Training Batch [{i+1}/{total_train_batches}]", end='\r')

                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                batch_size: int = inputs.size(0)

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_size
                total_train_samples += batch_size
                
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                correct += torch.sum(preds == labels)


            train_loss = running_loss
            train_loss /= total_train_samples
            train_accuracy = correct/total_train_samples
            training_losses.append(train_loss)
            training_accuracies.append(train_accuracy)

            model.eval()
            with torch.no_grad():

                TP = 0
                TN = 0
                FP = 0
                FN = 0

                for i, data in enumerate(val_loader):
                    print(f"Validating Batch [{i+1}/{total_val_batches}]", end='\r')

                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    batch_size = inputs.size(0)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * batch_size
                    total_val_samples += batch_size

                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    labels = labels.long().view(-1)

                    TP += ((preds == 1) & (labels == 1)).sum().item()
                    TN += ((preds == 0) & (labels == 0)).sum().item()
                    FP += ((preds == 1) & (labels == 0)).sum().item()
                    FN += ((preds == 0) & (labels == 1)).sum().item()

            val_loss /= total_val_samples
            val_accuracy = (TP + TN) / total_val_samples

            validation_losses.append(val_loss)
            validation_accuracies.append(val_accuracy)
            validation_cms.append((TP,TN,FP,FN))

            print(f"\n\ttraining_loss: {train_loss}\n\tvalidation_loss: {val_loss}")
            print(f"\n\ttraining_accuracy: {train_accuracy}\n\tvalidation_accuracy: {val_accuracy}")

            # early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        print(f"Creating Plots...", end='\r')
        plot_loss(training_losses, validation_losses, fold)
        plot_accuracy(training_accuracies, validation_accuracies, fold)
        plot_confusion_matrix(validation_cms, fold)
        plot_prf(validation_cms, fold)

        TP, TN, FP, FN = validation_cms[-1]
        accuracy = validation_accuracies[-1]
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Saving Model...", end='\r')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_loss': train_loss,
            'validation_loss': val_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }, models_dir / f"model_fold_{fold+1}.pth")

        print(f"Saved model on fold {fold+1}\n\ttraining_loss: {train_loss}\n\tvalidation_loss: {val_loss}")
