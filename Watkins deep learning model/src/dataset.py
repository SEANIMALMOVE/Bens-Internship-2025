import torch
from torch.utils.data import Dataset
import torchaudio
import os

### Get one torch sample from 1 spectogram

# Custom Dataset for loading spectrogram .pt files
class SpectrogramPTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_path):
                if fname.endswith(".pt"):
                    self.samples.append((
                        os.path.join(class_path, fname),
                        self.class_to_idx[class_name]
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        tensor = torch.load(path)  # loads spectrogram tensor

        # ensure tensor shape is [C, H, W]
        if tensor.dim() == 2:               # [H, W]
            tensor = tensor.unsqueeze(0)    # → [1, H, W]
        if tensor.dim() == 3 and tensor.shape[0] not in [1,3]:
            tensor = tensor.permute(2, 0, 1)

        # normalize manually (optional)
        tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-6)

        # ------------------------------------------------------------
        # FIX: Pad or crop all spectrograms to a fixed width
        TARGET_WIDTH = 400
        _, H, W = tensor.shape

        if W < TARGET_WIDTH:
            pad_amount = TARGET_WIDTH - W
            tensor = torch.nn.functional.pad(tensor, (0, pad_amount))
        else:
            tensor = tensor[:, :, :TARGET_WIDTH]
        # ------------------------------------------------------------

        if self.transform:
            tensor = self.transform(tensor)

        return tensor.float(), label