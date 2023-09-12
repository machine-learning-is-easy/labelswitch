from transformers import ViTConfig, ViTModel
import torch.nn as nn
# Initializing a ViT vit-base-patch16-224 style configuration

# Initializing a model (with random weights) from the vit-base-patch16-224 style configuration

class ViTForImageClassification(nn.Module):
    def __init__(self, num_labels, image_size):
        super(ViTForImageClassification, self).__init__()
        configuration = ViTConfig(num_hidden_layers=12, num_attention_heads=16, hidden_size=256, intermediate_size=128,
                                  encoder_stride=1, image_size=image_size, num_labels=num_labels)
        configuration.use_bfloat16 = True
        self.vit = ViTModel(configuration)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.rl = nn.ReLU()
        self.sf = nn.Softmax()
        self.num_labels = num_labels
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values, return_dict=True)
        logits = self.classifier(self.rl(outputs.pooler_output))
        logits = self.sf(self.rl(logits))
        return logits