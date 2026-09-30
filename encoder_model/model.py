import torch.nn as nn
from transformers import AutoModel

class DinoClassifier(nn.Module):
    def __init__(self, n_hid, n_classes, encoder_layer=None):
        super().__init__()
        if encoder_layer == None:
            self.encoder = AutoModel.from_pretrained("facebook/dinov2-small")
        else:
            self.encoder = encoder_layer

        self.head = nn.Linear(n_hid, n_classes)

    def forward(self, x):
        x = self.encoder(**x)
        cls = x.last_hidden_state[:, 0]
        return self.head(cls)