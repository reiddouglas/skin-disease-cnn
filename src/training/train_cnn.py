from cnn_models import *
import torch.optim as optim
from sklearn.model_selection import KFold
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import torch

k: int = 5
batch_size: int = 128
epochs: int = 100
patience: int = 5

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
kfold = KFold(n_splits=k)

# TODO: load data properly
filepath: str = '../data/data.npy'
train_dataset = np.load(file=filepath)

for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
    print(f"Fold {fold+1}/{k}")
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)

    # TODO: fill this out with hyperparameters
    model: CNN = CNN()
    model.to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)

    total_val_samples: int = 0
    total_train_samples: int = 0

    # early stopping variables
    best_val_loss: float = 0
    counter: int = 0

    model.train()
    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}]")
        running_loss: float = 0.0
        train_loss: float = 0.0
        val_loss: float = float('inf')

        for i, data in enumerate(train_loader):

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
            for _, data in enumerate(val_loader):

                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                batch_size: int = inputs.size(0)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * batch_size
                total_val_samples += batch_size
        
        val_loss /= total_val_samples

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_loss': train_loss,
            'validation_loss': val_loss
        }, f"models/model_fold_{fold+1}_epoch_{epoch+1}.pth")

        # early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered!")
                break 
