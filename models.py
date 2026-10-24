import torch
from torch import nn


class MLP(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )


class CNN(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )


class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patches = nn.Conv2d(1, 64, kernel_size=7, stride=7)
        self.positions = nn.Parameter(torch.randn(1, 16, 64) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=2,
            norm=nn.LayerNorm(64),
            enable_nested_tensor=False,
        )
        # Encoder layers are cloned, so initialize their weights independently.
        for parameter in self.encoder.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        self.classifier = nn.Linear(64, 10)

    def forward(self, images):
        tokens = self.patches(images).flatten(2).transpose(1, 2)
        encoded = self.encoder(tokens + self.positions)
        return self.classifier(encoded.mean(dim=1))


MODELS = {"mlp": MLP, "cnn": CNN, "cnn_aug": CNN, "transformer": VisionTransformer}
