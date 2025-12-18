"""
Baseline CNN training script

Training logic:
1. Forward pass: spectrogram -> model -> logits
2. Loss computation: compare logits with true labels
3. Backward pass: compute gradients
4. Optimizer step: update weights

Before training:
- logits are noisy
- predictions are random
- loss ~ log(num_classes)

After training:
- correct class logits increase
- wrong class logits decrease
- loss decreases
- accuracy improves

NOTE:
- Only model weights change
- Code and data stay the same
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm # type: ignore

# Import modules robustly so `py -m src.train` and `py src/train.py` both work
try:
    from .model import BaselineCNN
    from .dataloaders import get_dataloaders
except Exception:
    try:
        from model import BaselineCNN
        from dataloaders import get_dataloaders
    except Exception:
        from src.model import BaselineCNN
        from src.dataloaders import get_dataloaders


# =========================
# Training class
# =========================

class BaselineCNNTrainer:
    def __init__(
        self,
        spectrogram_dir: Path,
        checkpoint_path: Path,
        batch_size: int = 16,
        max_epochs: int = 10,
        patience: int = 3,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_path = checkpoint_path

        # -----------------------
        # Data
        # -----------------------
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            spectrogram_dir, batch_size=batch_size
        )

        self.num_classes = len(self.train_loader.dataset.classes)

        # -----------------------
        # Model
        # -----------------------
        self.model = BaselineCNN(
            input_channels=1,
            num_classes=self.num_classes
        ).to(self.device)

        # -----------------------
        # Training components
        # -----------------------
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # -----------------------
        # Early stopping state
        # -----------------------
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        print("Train samples:", len(self.train_loader.dataset), flush=True)
        print("Train batches:", len(self.train_loader), flush=True)


    # -----------------------
    # One training epoch
    # -----------------------
    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0

        desc = f"Epoch {epoch}/{self.max_epochs} Training"
        pbar = tqdm(
            self.train_loader,
            desc=desc,
            leave=True,
            ncols=100,
            unit="batch",
        )

        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}"
            })

        return running_loss / len(self.train_loader)
    # -----------------------
    # Validation
    # -----------------------
    def validate(self, epoch: int) -> float:
        self.model.eval()
        running_loss = 0.0

        desc = f"Epoch {epoch}/{self.max_epochs} Validation"
        pbar = tqdm(
            self.val_loader,
            desc=desc,
            leave=True,
            ncols=100,
            unit="batch",
        )

        with torch.no_grad():
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)

                running_loss += loss.item()

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}"
                })

        return running_loss / len(self.val_loader)


    # -----------------------
    # Save checkpoint
    # -----------------------
    def save_checkpoint(self):
        torch.save(self.model.state_dict(), self.checkpoint_path)

    # -----------------------
    # Full training loop
    # -----------------------

    def fit(self):
        for epoch in range(1, self.max_epochs + 1):
            print(f">>> epoch {epoch} started", flush=True)

            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)

            print(
                f"Epoch {epoch}/{self.max_epochs} "
                f"| Train Loss: {train_loss:.4f} "
                f"| Val Loss: {val_loss:.4f}",
                flush=True
            )


# =========================
# Entry point
# =========================

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    SPECT_DIR = PROJECT_ROOT / "Data" / "Spectrograms"
    CHECKPOINT_PATH = PROJECT_ROOT / "baseline_cnn_best.pth"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = BaselineCNNTrainer(
        spectrogram_dir=SPECT_DIR,
        checkpoint_path=CHECKPOINT_PATH,
        batch_size=16,
        max_epochs=10,
        patience=3,
        lr=1e-3,
        device=DEVICE,
    )

    trainer.fit()
