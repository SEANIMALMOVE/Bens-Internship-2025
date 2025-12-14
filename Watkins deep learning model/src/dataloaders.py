from torch.utils.data import DataLoader
from pathlib import Path
from src.dataset import SpectrogramPTDataset

### batching 16 torch samples

def get_dataloaders(
    spectrogram_root: Path,
    batch_size: int = 16,
):
    """
    Creates train / val / test dataloaders.

    spectrogram_root/
        train/
        val/
        test/
    """

    train_ds = SpectrogramPTDataset(spectrogram_root / "train")
    val_ds   = SpectrogramPTDataset(spectrogram_root / "val")
    test_ds  = SpectrogramPTDataset(spectrogram_root / "test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
