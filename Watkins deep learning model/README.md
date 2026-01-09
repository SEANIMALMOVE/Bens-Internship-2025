# First Deep Learning model: Audio Classification Using CNN and Transfer

## Project Overview

This project evaluates spectrogram-based convolutional neural networks for multiclass audio classification on the Watkins Marine Mammals dataset (44 species, 15,407 recordings). Audio recordings are converted to mel-spectrogram tensors (.pt), then used as image-like inputs for CNNs. We compare a compact Baseline CNN trained from scratch with a transfer-learning model based on EfficientNet‑B0.

## Motivation

Audio spectrograms can be treated as images and benefit from pretrained visual feature extractors. This project studies whether transfer learning (EfficientNet‑B0) improves generalization compared to a small custom CNN, and inspects model behaviour beyond accuracy using precision, recall and qualitative plots.

## Repository Structure
- PROJECT SHOWCASE IN src/Report.ipynb
- src/: training, model, dataloader and evaluation code.
- Data/: original audio and generated spectrograms.
  - Data/Annotations: audio_annotations.csv, train.csv, val.csv, test.csv.
  - Data/Spectrograms: train/, val/, test/ class folders containing .pt files.
  - notebooks/: Main.ipynb, Testing.ipynb, Report.ipynb, and exploration notebooks.
- outputs/:
  - analysis_plots/: gallery images, learning curves, stripplots.
  - histories/: saved training histories.
  - preds/, misclassified/, reports/: CSVs and JSON classification reports.
  - model checkpoints: baseline_best_2.pth, efficientnet_best_2.pth.

## Data & Preprocessing
- Dataset: 44 species, 15,407 recordings.
- Split: 70% train (10,784), 15% val (2,311), 15% test (2,312).
- Spectrograms saved as .pt files; each sample is converted to channel-first, padded/cropped to a fixed width (400 time frames), and normalized with per-sample standardization (offline).

### Models
- Baseline CNN: small conv stack (16→32→64→128) + global average pooling + classifier head; trained from scratch.
- EfficientNet‑B0: ImageNet-pretrained backbone adapted by repeating single-channel spectrograms to 3 channels and replacing the final linear layer. Backbone can be frozen/unfrozen for fine-tuning.

## Training
- Loss: CrossEntropyLoss with per-class weights to mitigate imbalance.
- Optimizer: AdamW, ReduceLROnPlateau scheduler.
- Early stopping on validation loss.
- Automatic Mixed Precision (AMP) supported on GPU.
- Typical runs: both models trained for up to 10 epochs.
  - Baseline: ≈3 min / epoch (+ ~20 s evaluation).
  - EfficientNet: ≈17 min / epoch (+ ~1 min evaluation).
  - End‑to‑end experiments took a couple of hours.

## Evaluation & Visualizations
- Quantitative: accuracy, training/validation loss, per-class precision and recall (weighted and macro). Numeric reports live in outputs/reports/.
- Qualitative:
  - Galleries: 25 lowest-confidence predictions (to inspect ambiguous/misclassified samples) and 25 random samples.
  - Learning curves: side-by-side accuracy and loss for both models.
  - Stripplots: per-sample prediction confidences (green = correct, red = incorrect).
  - Misclassification analysis and per-class confusion patterns available in notebooks/Main.ipynb and notebooks/Testing.ipynb.

## Key Findings
- EfficientNet converged to substantially higher validation accuracy and lower validation loss than the Baseline CNN under the chosen hyperparameters.
- Transfer learning produced steadier learning dynamics and improved precision/recall for most classes.
- Baseline CNN trains faster and is useful for quick prototyping or low-compute settings; EfficientNet offers better generalization at higher compute cost.
- Remaining high-confidence errors indicate systematic confusions or ambiguous spectrograms; class imbalance still affects rare species.

## Reproducibility
- Notebooks: notebooks/Main.ipynb, notebooks/Testing.ipynb, notebooks/Report.ipynb.
- Training pipeline: src/train.py; preprocessing pipeline in src/preprocess/.
- Results and figures: outputs/analysis_plots/, outputs/reports/, and saved checkpoints in outputs/.

## Technologies Used
- Python (3.8+)
- Jupyter / IPython
- PyTorch
- Torchvision
- Torchaudio
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- TQDM

## Summary
- EfficientNet‑B0 (transfer learning) clearly outperforms the Baseline CNN in accuracy, loss, precision and recall, at the cost of longer training time.
- All figures, numeric reports and artifacts are available in notebooks/ and outputs/ for inspection and reproduction.

## Conclusion

The results demonstrate that transfer learning with EfficientNet is highly effective for audio classification tasks based on spectrogram inputs. Even with a relatively simple training setup, pretrained models provide a strong performance advantage over custom CNNs trained from scratch.

This project highlights the importance of model selection and evaluation methodology in applied deep learning for audio analysis.

In the future it is optimal to try new models aswell!

