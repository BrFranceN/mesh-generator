
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


device = ("cuda"if torch.cuda.is_available() else "cpu")




class deformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.ff1 = nn.Sequential(
            nn.Linear
        )