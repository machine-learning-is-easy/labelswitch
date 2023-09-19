import torchvision
import torch.nn as nn
class Vit(nn.Module):
    def __init__(self, num_labels, image_size):
        super().__init__()
        self.network = torchvision.models.VisionTransformer(image_size=image_size, num_classes=num_labels,
                                                            patch_size=4, num_layers=12, num_heads=12,
                                                            hidden_dim=768, mlp_dim=3072)
    def forward(self, xb):
        return self.network(xb)