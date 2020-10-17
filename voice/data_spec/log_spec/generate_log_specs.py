# Script to generate log spectrograms from the audio files in data_clean. All
# spectrograms have the same duration, i.e., time_per_sample_breath or
# time_per_sample_cough, refer to set_audio_params.py

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
sys.path.append('../..')

import warnings
warnings.filterwarnings('ignore')

from set_audio_params import *
from spec_utils import generate_waveform, generate_spectrogram, save_spectrogram

def generate_log_spectrogram(waveform, sampling_rate, samples_per_frame, hop_length, window='hann'):
    """ Generates spectrogram with log scaled frequency axis and magnitude in decibels. """

    spec = generate_spectrogram(waveform, sampling_rate, samples_per_frame, hop_length, window=window)
    log_spec = librosa.amplitude_to_db(spec, ref=np.max)

    return log_spec

if __name__ == '__main__':
    # Currently audio types are breath and cough.
    audio_types = os.listdir('../../data_raw/data_clean')

    for audio_type in sorted(audio_types):
        # Overwriting spectrogram directory for audio type if it exists.
        if os.path.isdir('spec_' + audio_type):
            shutil.rmtree('spec_' + audio_type)
        os.makedirs(os.path.join('spec_' + audio_type, 'train'))
        os.makedirs(os.path.join('spec_' + audio_type, 'valid'))
        os.makedirs(os.path.join('spec_' + audio_type, 'test'))

        # Generating spectrograms for training audio samples.
        source_train = os.path.join('..', '..', 'data_raw', 'data_' + audio_type, 'train')
        dest_train = os.path.join('spec_' + audio_type, 'train')
        print(f'Generating log spectrograms for {audio_type}/train...')
        for dir in sorted(os.listdir(source_train)):
            os.makedirs(os.path.join(dest_train, dir))
            # For each class (asthma, covid, normal).
            for file in tqdm(sorted(os.listdir(os.path.join(source_train, dir))), desc=f'class={dir}'):
                input_path = os.path.join(source_train, dir, file)
                filename, extension = os.path.splitext(file)
                output_path = os.path.join(dest_train, dir, filename + '.png')

                if audio_type == 'breath':
                    waveform, timesteps = generate_waveform(input_path, sampling_rate, time_per_sample_breath)
                elif audio_type == 'cough':
                    waveform, timesteps = generate_waveform(input_path, sampling_rate, time_per_sample_cough)
                spec = generate_log_spectrogram(waveform, sampling_rate, samples_per_frame, hop_length, window='hann')
                save_spectrogram(spec, hop_length, output_path, x_axis='time', y_axis='log')

        # Generating spectrograms for validation audio samples.
        source_valid = os.path.join('..', '..', 'data_raw', 'data_' + audio_type, 'valid')
        dest_valid = os.path.join('spec_' + audio_type, 'valid')
        print(f'Generating log spectrograms for {audio_type}/valid...')
        for dir in sorted(os.listdir(source_valid)):
            os.makedirs(os.path.join(dest_valid, dir))
            # For each class (asthma, covid, normal).
            for file in tqdm(sorted(os.listdir(os.path.join(source_valid, dir))), desc=f'class={dir}'):
                input_path = os.path.join(source_valid, dir, file)
                filename, extension = os.path.splitext(file)
                output_path = os.path.join(dest_valid, dir, filename + '.png')

                if audio_type == 'breath':
                    waveform, timesteps = generate_waveform(input_path, sampling_rate, time_per_sample_breath)
                elif audio_type == 'cough':
                    waveform, timesteps = generate_waveform(input_path, sampling_rate, time_per_sample_cough)
                spec = generate_log_spectrogram(waveform, sampling_rate, samples_per_frame, hop_length, window='hann')
                save_spectrogram(spec, hop_length, output_path, x_axis='time', y_axis='log')

        # Generating spectrograms for test audio samples.
        source_test = os.path.join('..', '..', 'data_raw', 'data_' + audio_type, 'test')
        dest_test = os.path.join('spec_' + audio_type, 'test')
        print(f'Generating log spectrograms for {audio_type}/test...')
        for dir in sorted(os.listdir(source_test)):
            os.makedirs(os.path.join(dest_test, dir))
            # For each class (asthma, covid, normal).
            for file in tqdm(sorted(os.listdir(os.path.join(source_test, dir))), desc=f'class={dir}'):
                input_path = os.path.join(source_test, dir, file)
                filename, extension = os.path.splitext(file)
                output_path = os.path.join(dest_test, dir, filename + '.png')

                if audio_type == 'breath':
                    waveform, timesteps = generate_waveform(input_path, sampling_rate, time_per_sample_breath)
                elif audio_type == 'cough':
                    waveform, timesteps = generate_waveform(input_path, sampling_rate, time_per_sample_cough)
                spec = generate_log_spectrogram(waveform, sampling_rate, samples_per_frame, hop_length, window='hann')
                save_spectrogram(spec, hop_length, output_path, x_axis='time', y_axis='log')
