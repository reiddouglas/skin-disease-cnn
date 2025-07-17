import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import conv_output_size

class Filter():
    def __init__(self, size: int, stride: int, count: int, padding: int):
        super().__init__()
        self.size: int = size
        self.stride: int = stride
        self.count: int = count
        self.padding: int = padding

class Pool():
    def __init__(self, size: int, stride: int, padding: int):
        super().__init__()
        self.size: int = size
        self.stride: int = stride
        self.padding: int = padding

class FullyConnectedLayer():
    def __init__(self, output_neurons: int):
        super().__init__()
        self.output_neurons: int = output_neurons

class CNN(nn.Module):
    def __init__(self, input_channels: int, input_size: int, num_classes: int, filters: list[Filter], pools: list[Pool], fcls: list[FullyConnectedLayer], dropout: float = 0.3):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.convs: nn.ModuleList = nn.ModuleList()
        self.pools: nn.ModuleList = nn.ModuleList()
        self.fcs: nn.ModuleList = nn.ModuleList()

        current_channels: int = input_channels
        current_size: int = input_size

        for i in range(len(filters)):
            filter: Filter = filters[i]
            pool: Pool = pools[i]
            self.convs.append(nn.Conv2d(in_channels=current_channels,out_channels=filter.count,kernel_size=filter.size,stride=filter.stride,padding=filter.padding))
            current_size = conv_output_size(current_size, filter.padding, filter.size, filter.stride)
            self.pools.append(nn.MaxPool2d(kernel_size=pool.size,stride=pool.stride,padding=pool.padding))
            current_size = conv_output_size(current_size, pool.padding, pool.size, pool.stride)
            current_channels = filter.count
        
        prev_neurons: int = current_channels * (current_size ** 2)

        for layer in fcls:
            self.fcs.append(nn.Linear(in_features=prev_neurons,out_features=layer.output_neurons))
            prev_neurons = layer.output_neurons
        
        self.output_layer = nn.Linear(prev_neurons, num_classes)

        print(self.dropout, self.convs, self.pools, self.fcs)
    
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        """
        General structure of CNN: repeat(convolution layer -> pooling) -> flatten -> repeat(fully connected layer) -> output layer
        """
        for i in range(len(self.convs)):
            x = self.pools[i](F.relu(self.convs[i](x)))

        x = x.view(x.size(0), -1)

        for fc in self.fcs:
            x = F.relu(fc(x))
            x = self.dropout(x)


        # returns 'logits', NOT the final classification - pass into cross entropy loss during training
        x = self.output_layer(x)
        return x