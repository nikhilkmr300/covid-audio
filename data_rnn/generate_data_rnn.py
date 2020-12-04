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
sys.path.append(os.path.join('..', 'data_struc'))

from set_audio_params import *
from load_utils import *
from feature_extraction_utils import *

def generate_feature_matrix(waveform):
    """
    Generates feature matrix of shape [num_samples, num_timesteps, num_features]

    Parameters:
    waveform (NumPy array): Audio waveform.

    Returns:
    NumPy array (ndim=3): Feature matrix.
    """

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

    feature_matrix = np.zeros((num_timesteps, num_features))

    # Feature values are inserted into columns in the order provided in
    # rnn_instantaneous_features (refer set_audio_params.py).
    for i, feature in enumerate(rnn_instantaneous_features):
        if feature == 'rmse':
            rmse = rms_energy(waveform, samples_per_frame, hop_length)
            # Checking that length of the vector is equal to num_timesteps.
            assert len(rmse) == num_timesteps
            feature_matrix[:, i] = rmse.copy()
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

if __name__ == '__main__':
    # Types of audio (currently breath and cough).
    audio_types = os.listdir(os.path.join('..', 'data_raw', 'data_clean'))

    # MFCC features are instantaneous features.
    if 'mfcc' in struc_instantaneous_features:
        mfcc_features = ['mfcc' + str(i) for i in range(1, struc_n_mfcc + 1)]
        struc_instantaneous_features.extend(mfcc_features)

    # Removing the dummy literal 'mfcc' which stood for all coefficients from mfcc0
    # to mfcc<struc_n_mfcc>, as we have already handled the mfcc features above.
    struc_instantaneous_features.remove('mfcc')

    print(f'Using {len(struc_instantaneous_features)} instantaneous features (IF):')
    [print(instantaneous_feature, end='\t') for instantaneous_feature in struc_instantaneous_features]
    print()
    print()

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
        dest_train_X = os.path.join('data_' + audio_type, 'train_X.npy')
        dest_train_y = os.path.join('data_' + audio_type, 'train_y.npy')
        dest_valid_X = os.path.join('data_' + audio_type, 'valid_X.npy')
        dest_valid_y = os.path.join('data_' + audio_type, 'valid_y.npy')
        dest_test_X = os.path.join('data_' + audio_type, 'test_X.npy')
        dest_test_y = os.path.join('data_' + audio_type, 'test_y.npy')

        # Input to RNN should have shape [num_samples, num_timesteps, num_features].
        train_X = list()
        valid_X = list()
        test_X = list()

        # Targets corresponding to train, validation and test audio samples.
        train_y = list()
        valid_y = list()
        test_y = list()

        print(f'Generating train_X.npy and train_y.npy for audio_type={audio_type}...')
        for i, dir in enumerate(sorted(os.listdir(source_train))):
            dir = os.path.join(source_train, dir)
            class_ = os.path.basename(dir)

            # Ignoring class asthma.
            if ignore_asthma == True and class_ == 'asthma':
                print('Ignoring class=asthma.')
                continue

            # Adding targets right here and not in the nested for loop below as we
            # are iterating classwise anyway.
            if class_ == 'covid':
                train_y.extend([1] * len(os.listdir(dir)))
            elif class_ == 'asthma' or class_ == 'normal':
                train_y.extend([0] * len(os.listdir(dir)))

            for file in tqdm(sorted(os.listdir(dir)), desc=f'class={class_}'):
                file = os.path.join(dir, file)

                if audio_type == 'breath':
                    waveform, _ = load(file, sampling_rate, time_per_sample_breath)
                elif audio_type == 'cough':
                    waveform, _ = load(file, sampling_rate, time_per_sample_cough)

                feature_matrix = generate_feature_matrix(waveform)
                train_X.append(feature_matrix)

        train_X = np.array(train_X)
        train_y = np.array(train_y)

        # Sanity check that number of samples in feature matrix and target vector
        # are the same.
        assert train_X.shape[0] == train_y.shape[0]

        np.save(dest_train_X, train_X)
        np.save(dest_train_y, train_y)

        print(f'Generating valid_X.npy and valid_y.npy for audio_type={audio_type}...')
        for i, dir in enumerate(sorted(os.listdir(source_valid))):
            dir = os.path.join(source_valid, dir)
            class_ = os.path.basename(dir)

            # Ignoring class asthma.
            if ignore_asthma == True and class_ == 'asthma':
                print('Ignoring class=asthma.')
                continue

            # Adding targets right here and not in the nested for loop below as we
            # are iterating classwise anyway.
            if class_ == 'covid':
                valid_y.extend([1] * len(os.listdir(dir)))
            elif class_ == 'asthma' or class_ == 'normal':
                valid_y.extend([0] * len(os.listdir(dir)))

            for file in tqdm(sorted(os.listdir(dir)), desc=f'class={class_}'):
                file = os.path.join(dir, file)

                if audio_type == 'breath':
                    waveform, _ = load(file, sampling_rate, time_per_sample_breath)
                elif audio_type == 'cough':
                    waveform, _ = load(file, sampling_rate, time_per_sample_cough)

                feature_matrix = generate_feature_matrix(waveform)
                valid_X.append(feature_matrix)

        valid_X = np.array(valid_X)
        valid_y = np.array(valid_y)

        # Sanity check that number of samples in feature matrix and target vector
        # are the same.
        assert valid_X.shape[0] == valid_y.shape[0]

        np.save(dest_valid_X, valid_X)
        np.save(dest_valid_y, valid_y)

        print(f'Generating test_X.npy and test_y for audio_type={audio_type}...')
        for i, dir in enumerate(sorted(os.listdir(source_test))):
            dir = os.path.join(source_test, dir)
            class_ = os.path.basename(dir)

            # Ignoring class asthma.
            if ignore_asthma == True and class_ == 'asthma':
                print('Ignoring class=asthma.')
                continue

            # Adding targets right here and not in the nested for loop below as we
            # are iterating classwise anyway.
            if class_ == 'covid':
                test_y.extend([1] * len(os.listdir(dir)))
            elif class_ == 'asthma' or class_ == 'normal':
                test_y.extend([0] * len(os.listdir(dir)))

            for file in tqdm(sorted(os.listdir(dir)), desc=f'class={class_}'):
                file = os.path.join(dir, file)

                if audio_type == 'breath':
                    waveform, _ = load(file, sampling_rate, time_per_sample_breath)
                elif audio_type == 'cough':
                    waveform, _ = load(file, sampling_rate, time_per_sample_cough)

                feature_matrix = generate_feature_matrix(waveform)
                test_X.append(feature_matrix)

        test_X = np.array(test_X)
        test_y = np.array(test_y)

        # Sanity check that number of samples in feature matrix and target vector
        # are the same.
        assert valid_X.shape[0] == valid_y.shape[0]

        np.save(dest_test_X, test_X)
        np.save(dest_test_y, test_y)
