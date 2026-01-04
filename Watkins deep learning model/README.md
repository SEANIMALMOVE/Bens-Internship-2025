# First Deep Learning model: Audio Classification Using CNN and Transfer 

## Project Overview

This project investigates the application of deep learning models for audio classification using mel-spectrogram representations. The primary objective is to compare a custom baseline Convolutional Neural Network (CNN) with a transfer learning–based EfficientNet model, evaluating their performance on a multiclass audio classification task. We were working with 44 different species classes.

After dividing into test/train/val, audio signals are then converted into spectrogram tensors and stored as .pt files. These spectrograms are then used as image-like inputs for convolutional neural networks. The project focuses on model design, training stability, and rigorous evaluation using standard classification metrics.

## Motivation

Audio classification problems often suffer from limited labeled data and high variability in acoustic patterns. While custom CNNs can learn task-specific features, they may struggle to generalize. Transfer learning offers a powerful alternative by leveraging pretrained visual feature extractors, even when applied to non-visual domains such as audio spectrograms.

This project explores:
- Whether a simple baseline CNN is sufficient for the task
- How much performance gain can be achieved using a pretrained EfficientNet
- How evaluation metrics beyond accuracy reveal differences in model behavior

## Folder Structure
- Watkins deep learning model -> main folder
- src folder within main folder: python codes
- Data/Annotations: annotations file (Audio and Spectrogram files are kept here locally)
- notebooks: 
  - Main codelines where I run the project in the Main notebook
  - Report notebook: Project description, summary and report
- outputs: results after evaluation
    - analysis_plots: evaluation charts and results
    - histories: all previous training histories (for each version)
    - missclassified: missclassified classes
    - preds: all predictions and actual values. Confidience of predictions
    - reports: precision, recall, f1-score for each class (44 classes)

## Models Implemented
### Baseline CNN
- Custom-designed convolutional network
- Four convolutional blocks with ReLU and max pooling
- Global Average Pooling to reduce overfitting
- Fully connected classifier head
- Trained from scratch on spectrogram data

### EfficientNet-B0 (Transfer Learning)
- Pretrained on ImageNet
- Adapted for single-channel spectrogram input by channel replication
- Fine-tuned on the target dataset
- Significantly higher performance compared to the baseline model

## Dataset and Preprocessing
- Input data consists of mel-spectrograms saved as PyTorch tensors
- Directory structure:

Spectrograms/
  train/
  val/
  test/


- All spectrograms are padded or cropped to a fixed width for batch consistency
- Offline normalization is applied

## Training Setup
- Loss function: CrossEntropyLoss with class balancing
- Optimizer: AdamW
- Learning rate scheduling: ReduceLROnPlateau
- Early stopping based on validation loss
- Automatic Mixed Precision (AMP) enabled on GPU

## Evaluation
Models are evaluated exclusively on the test set using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

This ensures an unbiased comparison between the baseline CNN and the EfficientNet model.

## Key Results
- The baseline CNN achieves moderate performance and shows signs of underfitting
- EfficientNet significantly outperforms the baseline across all evaluation metrics in this experimental result
- Confusion matrices indicate a substantial reduction in misclassifications when using transfer learning

## Technologies Used
- Python
- PyTorch
- Torchvision
- Torchaudio
- Scikit-learn
- NumPy
- TQDM

## Conclusion

The results demonstrate that transfer learning with EfficientNet is highly effective for audio classification tasks based on spectrogram inputs. Even with a relatively simple training setup, pretrained models provide a strong performance advantage over custom CNNs trained from scratch.

This project highlights the importance of model selection and evaluation methodology in applied deep learning for audio analysis.

In the future it is optimal to try new models aswell!

