from pathlib import Path
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import torch

# generate spectograms normalization

def generate_spectrogram(audio_path, output_path, target_sr=44000, n_mels=128):
    # Loads audio, normalizes, converts to mel-spectrogram, saves to .pt

    waveform, sr = torchaudio.load(audio_path)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    # Normalize waveform to [-1, 1]
    waveform = waveform / waveform.abs().max()

    # Mel spectrogram
    mel = MelSpectrogram(sample_rate=target_sr, n_mels=n_mels)(waveform)
    mel_db = AmplitudeToDB()(mel)

    # Save
    output_path = Path(output_path)
    torch.save(mel_db, output_path)

    return output_path

