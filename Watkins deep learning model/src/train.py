## 1. forward pass: spectogram -> model -> logits
## 2. loss function: given the logits and the true label how bad is the prediction?
## 3. backward pass: compute gradients: which weights caused this mistake, and in what direction should they change
## 4. optimizer step: update weights based on gradients and learning rate just a tiny bit

# before training: outputs are noisy, logits are small and similar, predictions are random
# after training: output correct class logits become larger, wrong class logits become smaller, loss decreases, accuracy increases (shape of tensors dont change, only the meaning of the numbers)


import torch
import torch.nn as nn
import torch.optim as optim
from src.model import BaselineCNN
