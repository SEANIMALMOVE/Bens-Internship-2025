import torch
from torch.utils.data import Dataset
import os  
from src.dataset import SpectrogramPTDataset
from pathlib import Path

from torch.utils.data import DataLoader

ROOT_PATH = Path("..")  

train_ds = SpectrogramPTDataset(ROOT_PATH/"Data/Spectrograms/train")
val_ds   = SpectrogramPTDataset(ROOT_PATH/"Data/Spectrograms/val")
test_ds  = SpectrogramPTDataset(ROOT_PATH/"Data/Spectrograms/test")

batch_size = 16

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

print("Train size:", len(train_ds))
print("Val size:  ", len(val_ds))
print("Test size: ", len(test_ds))
# print("Shape example:", next(iter(train_loader))[0].shape)