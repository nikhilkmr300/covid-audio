# Script to preprocess data_clean to generate spectrograms.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from tqdm import tqdm
import shutil
import os

import sys
sys.path.append('..')

import warnings
warnings.filterwarnings('ignore')

from spec_utils import generate_waveform, generate_spectrogram, save_spectrogram

def generate_mel_spectrogram(waveform, sampling_rate, samples_per_frame, hop_length, n_mels=13, window='hann'):
    spec = generate_spectrogram(waveform, sampling_rate, samples_per_frame, hop_length, window=window)
    mel_spec = librosa.feature.melspectrogram(S=spec, sr=sampling_rate, n_fft=samples_per_frame, hop_length=hop_length, n_mels=n_mels)

    return mel_spec

sampling_rate = 10000
samples_per_frame = 512
hop_length = samples_per_frame // 4

print('Parameters for spectrograms (you can change these in generate_mel_specs.py):')
print(f'\tsampling_rate={sampling_rate}Hz')
print(f'\tsamples_per_frame={samples_per_frame}')
print(f'\thop_length={hop_length}')

# Currently audio types are breath and cough.
audio_types = os.listdir('../../data_raw/data_clean')

for audio_type in (audio_types):
    # Overwriting spectrogram directory for audio type if it exists.
    if os.path.isdir('spec_' + audio_type):
        shutil.rmtree('spec_' + audio_type)
    os.makedirs(os.path.join('spec_' + audio_type, 'train'))
    os.makedirs(os.path.join('spec_' + audio_type, 'valid'))
    os.makedirs(os.path.join('spec_' + audio_type, 'test'))

    # Generating spectrograms for training audio samples.
    source_train = os.path.join('..', '..', 'data_raw', 'data_' + audio_type, 'train')
    dest_train = os.path.join('spec_' + audio_type, 'train')
    print(f'Generating spectrograms for {audio_type}/train...')
    for dir in os.listdir(source_train):
        os.makedirs(os.path.join(dest_train, dir))
        # For each class (asthma, covid, normal).
        for file in tqdm(os.listdir(os.path.join(source_train, dir)), desc=f'class={dir}'):
            input_path = os.path.join(source_train, dir, file)
            filename, extension = os.path.splitext(file)
            output_path = os.path.join(dest_train, dir, filename + '.png')
            waveform, timesteps = generate_waveform(input_path, sampling_rate=sampling_rate)
            spec = generate_mel_spectrogram(waveform, sampling_rate, samples_per_frame, hop_length, window='hann')
            save_spectrogram(spec, hop_length, output_path, x_axis='time', y_axis='mel')

    # Generating spectrograms for validation audio samples.
    source_valid = os.path.join('..', '..', 'data_raw', 'data_' + audio_type, 'valid')
    dest_valid = os.path.join('spec_' + audio_type, 'valid')
    print(f'Generating spectrograms for {audio_type}/valid...')
    for dir in os.listdir(source_valid):
        os.makedirs(os.path.join(dest_valid, dir))
        # For each class (asthma, covid, normal).
        for file in tqdm(os.listdir(os.path.join(source_valid, dir)), desc=f'class={dir}'):
            input_path = os.path.join(source_valid, dir, file)
            filename, extension = os.path.splitext(file)
            output_path = os.path.join(dest_valid, dir, filename + '.png')
            waveform, timesteps = generate_waveform(input_path, sampling_rate=sampling_rate)
            spec = generate_mel_spectrogram(waveform, sampling_rate, samples_per_frame, hop_length, window='hann')
            save_spectrogram(spec, hop_length, output_path, x_axis='time', y_axis='mel')

    # Generating spectrograms for test audio samples.
    source_test = os.path.join('..', '..', 'data_raw', 'data_' + audio_type, 'test')
    dest_test = os.path.join('spec_' + audio_type, 'test')
    print(f'Generating spectrograms for {audio_type}/test...')
    for dir in os.listdir(source_test):
        os.makedirs(os.path.join(dest_test, dir))
        # For each class (asthma, covid, normal).
        for file in tqdm(os.listdir(os.path.join(source_test, dir)), desc=f'class={dir}'):
            input_path = os.path.join(source_test, dir, file)
            filename, extension = os.path.splitext(file)
            output_path = os.path.join(dest_test, dir, filename + '.png')
            waveform, timesteps = generate_waveform(input_path, sampling_rate=sampling_rate)
            spec = generate_mel_spectrogram(waveform, sampling_rate, samples_per_frame, hop_length, window='hann')
            save_spectrogram(spec, hop_length, output_path, x_axis='time', y_axis='mel')
