from glob import glob
import numpy as np
import os
import torch
import torchaudio
from torch import nn
torchaudio.set_audio_backend("sox_io")


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, paths, params, transform=None):
        super().__init__()
        self.filenames = []
        self.transform = transform
        self.audio_length = params['audio_length']
        for path in paths:
            self.filenames += glob(f'{path}/**/*.wav', recursive=True)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        signal, _ = torchaudio.load(audio_filename)
        if self.transform is not None:
            signal = self.transform(signal)

        # renormalize the audio
        scaler = max(signal.max(), -signal.min())
        if scaler > 0:
            signal = signal / scaler
        return {
            'audio': signal
        }

# Pad and crop the audio sample
class PadCrop(nn.Module):
    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

    def __call__(self, input):
        n, s = input.shape
        #Always start at the beginning of the sample
        start = 0 #torch.randint(0, max(0, s - self.n_samples) + 1, []).item()
        end = start + self.n_samples
        output = input.new_zeros([n, self.n_samples])
        output[:, :min(s, self.n_samples)] = input[:, start:end]
        return output

def from_path(data_dirs, params):
    dataset = AudioDataset(data_dirs, params, transform=PadCrop(params['audio_length']))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params['batch_size'],
        collate_fn=None,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        drop_last=True)
