import torchvision
import torch.nn as nn
class ResNet(nn.Module):
    def __init__(self, num_label):
        super().__init__()
        self.network = torchvision.models.resnet34()
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, out_features=num_label)

    def forward(self, xb):
        return self.network(xb)