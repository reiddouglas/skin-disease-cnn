from cnn_models import *
from utils import *
import torch.optim as optim
from sklearn.model_selection import KFold
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset, Dataset
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
import numpy as np
import torch
from collections import Counter


# custom dataset to ensure not all images need to be loaded into memory at once
class NPZBatchDataset(Dataset):
    def __init__(self, npz_file_paths, transform=None):
        self.npz_file_paths = npz_file_paths
        self.transform = transform
        
        self.file_sizes = []
        for path in npz_file_paths:
            with np.load(training_dir / path) as data:
                self.file_sizes.append(len(data['target']))
        self.total_size = sum(self.file_sizes)
        
        self.index_mapping = []
        for file_idx, size in enumerate(self.file_sizes):
            for local_idx in range(size):
                self.index_mapping.append( (file_idx, local_idx) )
            
        self._cache_file_idx = None
        self._cache_data = None

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        file_idx, local_idx = self.index_mapping[idx]
        file_path = self.npz_file_paths[file_idx]

        if self._cache_file_idx != file_idx:
            self._cache_data = np.load(training_dir / file_path)
            self._cache_file_idx = file_idx
        
        data = self._cache_data
        image = data['image'][local_idx]
        label = data['target'][local_idx]

        if self.transform:
            image = self.transform(torch.tensor(image, dtype=torch.float32))

        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label

# file path constants
script_dir = Path(__file__).parent.resolve()
models_dir = script_dir / 'models'
training_dir = script_dir / 'training_data'
models_dir.mkdir(exist_ok=True)

# training hyperparameters
k: int = 5
kfold = KFold(n_splits=k)
batch_size: int = 32
epochs: int = 20
patience: int = 5

# try to use CUDA (graphics card)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print(f"Using {device}")

# load files into the dataset
files = get_filenames(training_dir, ext='.npz')
train_dataset = NPZBatchDataset(files)
classes = 2

for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)

    total_train_batches = len(train_loader)
    total_val_batches = len(val_loader)

    model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier[0] = nn.Dropout(p=0.3)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, classes)
    model.to(device=device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)

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

        train_loss = running_loss
        train_loss /= total_train_samples

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                print(f"Validating Batch [{i+1}/{total_val_batches}]", end='\r')

                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                batch_size: int = inputs.size(0)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * batch_size
                total_val_samples += batch_size
        
        val_loss /= total_val_samples

        print(f"\ttraining_loss: {train_loss}\n\tvalidation_loss: {val_loss}")

        # early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered!")
                break 
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_loss': train_loss,
        'validation_loss': val_loss
    }, script_dir / f"models/model_fold_{fold+1}_epoch_{epoch+1}.pth")

    print(f"Saved model on fold {fold+1}\n\ttraining_loss: {train_loss}\n\tvalidation_loss: {val_loss}")
