from torch.utils.data import DataLoader
from pathlib import Path

# Import the dataset in a way that works when running as a script
# or as a package. Try package-relative first, then fall back.
try:
    from .dataset import SpectrogramPTDataset
except Exception:
    try:
        from dataset import SpectrogramPTDataset
    except Exception:
        from src.dataset import SpectrogramPTDataset

### batching 16 torch samples

def get_dataloaders(
    spectrogram_root: Path,
    batch_size: int = 64,
    num_workers: int = 4,
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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
