from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.projector = torch.nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.top_linear =  nn.Linear(32, num_classes)

    def forward(self, features):
        features = self.projector(features)
        outputs = self.top_linear(features)

        return outputs