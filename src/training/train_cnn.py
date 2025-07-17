from cnn_models import *
from utils import *
import torch.optim as optim
from sklearn.model_selection import KFold
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import numpy as np
import torch

script_dir = Path(__file__).parent.resolve()
models_dir = script_dir / 'models'
models_dir.mkdir(exist_ok=True)

k: int = 5
batch_size: int = 32
epochs: int = 20
patience: int = 5

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print(f"Using {device}")
kfold = KFold(n_splits=k)

dir = 'training_data'

images = get_images_from_file(dir)
images = images.transpose(0,3,1,2)
tensor_x = torch.Tensor(images)
# TODO: replace with real labels
tensor_y = torch.LongTensor(np.random.randint(2, size=(images.shape[0])))

train_dataset = TensorDataset(tensor_x, tensor_y)

for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
    print(f"Fold {fold+1}/{k}")
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)

    # TODO replace with real class count
    classes = 2
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

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
        print(f"Epoch [{epoch+1}/{epochs}]")
        running_loss: float = 0.0
        train_loss: float = 0.0
        val_loss: float = 0.0

        model.train()
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

        print(f"Model on fold {fold+1}, epoch {epoch+1}\n\ttraining_loss: {train_loss}\n\tvalidation_loss: {val_loss}")

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
