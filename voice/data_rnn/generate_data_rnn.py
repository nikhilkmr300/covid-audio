# Script to generate train, valid and test arrays of the shape desired by
# Keras RNNs, so that they can be fed into the network.

import numpy as np
from tqdm import tqdm
import shutil
import os

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
sys.path.append('../data_struc')

from set_audio_params import *
from load_utils import *
from feature_extraction_utils import *

def generate_feature_matrix(waveform):
    # Output after framing has shape:
    # floor((audio_length + 2 * pad_length - frame_length) / hop_length) + 1
    # Librosa takes pad_length = floor(frame_length / 2).
    pad_length = samples_per_frame // 2
    num_timesteps = (waveform.shape[0] + 2 * pad_length - samples_per_frame) // hop_length + 1

    # Calculating num_features, handling MFCCs separately.
    if 'mfcc' in rnn_instantaneous_features:
        num_features = len(rnn_instantaneous_features) - 1
        num_features += rnn_n_mfcc      # Each coefficient is a feature
    else:
        num_features = len(rnn_instantaneous_features)

    # Feature values are inserted into columns in the order provided in
    # rnn_instantaneous_features (refer set_audio_params.py).
    for i, feature in enumerate(rnn_instantaneous_features):
        feature_matrix = np.zeros((num_timesteps, num_features))

        if feature == 'rmse':
            rmse = rms_energy(waveform, samples_per_frame, hop_length)
            # Checking that length of the vector is equal to num_timesteps.
            assert len(rmse) == num_timesteps
            feature_matrix[:, i] = rmse
        elif feature == 'zcr':
            zcr = zero_crossing_rate(waveform, samples_per_frame, hop_length)
            # Checking that length of the vector is equal to num_timesteps.
            assert len(zcr) == num_timesteps
            feature_matrix[:, i] = zcr
        elif feature == 'sc':
            sc = spectral_centroid(waveform, sampling_rate, samples_per_frame, hop_length)
            # Checking that length of the vector is equal to num_timesteps.
            assert len(sc) == num_timesteps
            feature_matrix[:, i] = sc
        elif feature == 'sb':
            sb = spectral_bandwidth(waveform, sampling_rate, samples_per_frame, hop_length)
            # Checking that length of the vector is equal to num_timesteps.
            assert len(sb) == num_timesteps
            feature_matrix[:, i] = sb
        elif feature == 'sr':
            sr = spectral_rolloff(waveform, sampling_rate, samples_per_frame, hop_length, rnn_roll_percent)
            # Checking that length of the vector is equal to num_timesteps.
            assert len(sr) == num_timesteps
            feature_matrix[:, i] = sr
        elif feature == 'mfcc':
            mfcc_matrix = mfcc(waveform, sampling_rate, samples_per_frame, hop_length, rnn_n_mfcc)
            # mfcc_matrix has shape [n_mfcc, num_timesteps], taking
            # transpose.
            mfcc_matrix = mfcc_matrix.T
            feature_matrix[:, i:i+rnn_n_mfcc] = mfcc_matrix

    # Checking that feature_matrix has the correct shape.
    assert feature_matrix.shape == (num_timesteps, num_features)

    return feature_matrix

# Types of audio (currently breath and cough).
audio_types = os.listdir(os.path.join('..', 'data_raw', 'data_clean'))

# Making directories for each audio type.
for audio_type in audio_types:
    if os.path.exists('data_' + audio_type):
        shutil.rmtree('data_' + audio_type)
    os.makedirs(os.path.join('data_' + audio_type))

for audio_type in sorted(audio_types):
    # Paths to train, valid and test directories for each audio type.
    source_train = os.path.join('..', 'data_raw', 'data_' + audio_type, 'train')
    source_valid = os.path.join('..', 'data_raw', 'data_' + audio_type, 'valid')
    source_test = os.path.join('..', 'data_raw', 'data_' + audio_type, 'test')
    # Paths to destination files.
    dest_train = os.path.join('data_' + audio_type, 'train.npy')
    dest_valid = os.path.join('data_' + audio_type, 'valid.npy')
    dest_test = os.path.join('data_' + audio_type, 'test.npy')

    # Input to RNN should have shape [num_samples, num_timesteps, num_features].
    train = list()
    valid = list()
    test = list()

    print(f'Generating train.npy for audio_type={audio_type}...')
    for i, dir in enumerate(sorted(os.listdir(source_train))):
        dir = os.path.join(source_train, dir)
        class_ = os.path.basename(dir)

        for file in tqdm(sorted(os.listdir(dir)), desc=f'class={class_}'):
            file = os.path.join(dir, file)

            if audio_type == 'breath':
                waveform, _ = load(file, sampling_rate, time_per_sample_breath)
            elif audio_type == 'cough':
                waveform, _ = load(file, sampling_rate, time_per_sample_cough)

            feature_matrix = generate_feature_matrix(waveform)
            train.append(feature_matrix)

    train = np.array(train)
    np.save(dest_train, train)

    print(f'Generating valid.npy for audio_type={audio_type}...')
    for i, dir in enumerate(sorted(os.listdir(source_valid))):
        dir = os.path.join(source_valid, dir)
        class_ = os.path.basename(dir)

        for file in tqdm(sorted(os.listdir(dir)), desc=f'class={class_}'):
            file = os.path.join(dir, file)

            if audio_type == 'breath':
                waveform, _ = load(file, sampling_rate, time_per_sample_breath)
            elif audio_type == 'cough':
                waveform, _ = load(file, sampling_rate, time_per_sample_cough)

            feature_matrix = generate_feature_matrix(waveform)
            valid.append(feature_matrix)

    valid = np.array(valid)
    np.save(dest_valid, valid)

    print(f'Generating test.npy for audio_type={audio_type}...')
    for i, dir in enumerate(sorted(os.listdir(source_test))):
        dir = os.path.join(source_test, dir)
        class_ = os.path.basename(dir)

        for file in tqdm(sorted(os.listdir(dir)), desc=f'class={class_}'):
            file = os.path.join(dir, file)

            if audio_type == 'breath':
                waveform, _ = load(file, sampling_rate, time_per_sample_breath)
            elif audio_type == 'cough':
                waveform, _ = load(file, sampling_rate, time_per_sample_cough)

            feature_matrix = generate_feature_matrix(waveform)
            test.append(feature_matrix)

    test = np.array(test)
    np.save(dest_test, test)
