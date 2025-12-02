# Review of Ecoacoustic Deep Learning Models Relevant to Marine Species Classification 
**Purpose:** Summary of key ecoacoustic research articles and explanation of how their methods inform the design of a neural network for marine species sound classification.

---

## 1. Introduction
In this file I'll explain what I've read in articles the past week. Some info may be unrelevant, some are useful, but it's better to collect everything we can.
---

## 2. Paper Summaries

### Acoustic fish species identification using deep learning and machine learning algorithms: A systematic review

Passive Acoustic Monitoring (PAM) is used to study biodiversity by recording animal vocalizations over long periods. The 2023 ScienceDirect review evaluates how deep learning has transformed PAM across birds, mammals, insects, amphibians, and marine ecosystems.

- which deep learning architectures work best
- why Mel spectrograms dominate ecoacoustics
- how to deal with noise, overlaps, and small datasets
- how to design and train a robust neural network

Spectograms:
- Deep learning models almost always use spectrograms instead of raw audio-
- reason: Spectrograms capture harmonics

Architectures:
- CNNs (Convolutional Neural Networks)
    - Detect 2D patterns in spectrograms
    - Excellent for harmonic or tonal signals
    - Efficient and easy to train
    - focus on analyzing the eintire image to assign a single label
    - good baseline

- CRNNs (CNN + LSTM/GRU)
    - Add temporal modeling
    - Useful for longer whale calls, pulse trains, songs
    - ideal for sequiental data
    - identifies multiple object withing regions of the image
    - we could use if the temporal structural matters, but for detection I don't think so

for marine species CNN or CRNN is ideal

Segmentation:
- typically 1-5s 
- makes training easier
- localizes events
- extracts multiple training samples
- reduces memory usage

Data challanges & solutions:
- Noise augmentation (adding random noise to make the model more "real-life" scenario)
- Filtering (filter unwanted frequencies: low frequency boats)
- Spectral subtraction (removes background noise from spectogram)
- Robust CNN layers (resist noise, distortions, preventing overfitting to clean signals, )

- Polyphony (overlapping species):
    - Sigmoid outputs instead of softmax
    - Multi-label training

- Solutions:
    - Weighted loss ( give a higher penalty (weight) to mistakes on rare classes, and a lower penalty to mistakes on common classes)
    - Oversampling of minority classes (Taking the classes (species) with very few recordings and repeating them more often during training)
    - Synthetic augmentation

Lack of labeled data: (review new techniques)
- Semi-supervised learning
- Self-supervised learning
- Transfer learning (reducing the amount of data needed to train a CNN)
- Pretrained audio models

General steps for PAM systems:
- Audio acquisition
- Segmentation
- Spectogram generation
- Deep learning inference (NN takes the spectrogram and makes predictions)
- Post-processing (After the model produces raw probabilities, we apply additional rules to clean and improve the predictions)
- Species level decision(taking the processed probabilities and making the final decision about what species is present in each segment: deciding which species exceed confidence level)

### BirdNet

It is the most widely used deep learning system in ecoacoustics. Although BirdNET is built for birds, we can still use it for marine animal acoustics.

more than 50,000 species
Millions of annotated recordings

It converts audio segments into log-Mel spectograms:
- n_mels = 128
- Window size ≈ 25 ms
- Hop length ≈ 10 ms
- Audios are usually 3s long

Although BirdNet uses huge number of species, mine will have to use less.

---
BirdNET uses a custom deep convolutional neural network inspired by:
- ResNet-like residual blocks (as layers increase model becomes harder to optimize: kindof a shortcut connection from input to output) (idk if its neccesarry)
- EfficientNet scaling principles (To make a model bigger, you should increase three things together in balanced proportions:
    - Depth → more layers
    - Width → more filters per layer
    - Resolution → larger input images (spectrograms))

Bird audio datasets often contain:
- Background species
- Overlapping sounds
- Human noise
- Inconsistent labels

BirdNET handles this by:
- Using noise augmentation during training (adding random noises to the dataset to "mimic" real world imperfection)
- Using data imbalance corrections (the neural network will always prefer the species with more data, because these dominate training)
- Allowing “weak labels” (only main species labeled)

BirdNET outputs:
- probability vector for all species -> probabilities reflect likelihood of presence
- sigmoid or softmax output (choose 1 or any number of species to turn the model's raw numbers into probabilities. at the end of the NN)
- Threshold tuning will be required (every species will get a probability between 0 and 1. diff species require diff thresholds)
- Class imbalance can be handled via weighted loss (some classes have much more data than others. Weight loss gives a larger penalty to mistakes on rare species and smaller penalty on common species)

Transfer Learning: 
- take a neural network that was already trained on a huge dataset
- keep all the knowledge it learned about sound patterns
- only retrain the last small part of the network on your own dataset
It would be easier to reuse the already learnt big network for my own model
for example we could reuse the convolutional layer (universal sound patterns, harmonics, frequency). We have a small dataset, we dont have millions of labeled underwater recordings. Also dont need a huge CNN.
retrain: hidden layers and output layer

Birdnet Step-by-step:
1. Load raw audio
2. Resample to 48 kHz or 32 kHz
3. Segment into 3-second windows
4. Compute log-Mel spectrogram
5. Normalize across frequency bins
6. Forward pass through CNN
7. Output probability vector

Strength of BirdNet
- High accuracy in noisy enviroment
- Works good w many species
- Efficient CNN architecture
- Optimized spectrogram processing
- Transfer learning support

### Perch

PERCH is a deep learning system created by Google Research to solve a very hard ecoacoustics problem: When multiple species call at the same time in the same audio.

Most standard classifiers (CNN + softmax) assume:
- only one species per audio segment

But in nature:
- two birds can sing at once
- a bird + insect + wind noise happen together
- marine animals overlap too (dolphins + whales + boat noise)
- BirdNET struggles with very strong overlaps, even though it's good overall.

PERCH solves this. But in out Watking Marine Mammals database there is no overlap. It's only one animal at a time.

Permutation-Invariant Training (PIT):
- detect any number of species
- any order

Source separation embedding:
- PERCH tries to separate overlapping sounds even before classification
- It learns a latent embedding space (A space where the neural network represents sounds as numbers that capture their meaning, similarity, and structure)
- sounds that belong to the same “source” cluster together
- 

Classification. PERCH uses:
- multiple outputs
- sigmoid activation
- multi-label detection