import numpy as np
import pandas as pd
import librosa
import itertools
from tqdm import tqdm
import shutil
import os

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('../eda')

from feature_extraction_utils import *

# Types of audio (currently breath and cough).
audio_types = os.listdir(os.path.join('..', 'data_raw', 'data_clean'))

# Making train, valid and test directories for each audio type.
for audio_type in audio_types:
    if os.path.exists('data_' + audio_type):
        shutil.rmtree('data_' + audio_type)
    os.makedirs(os.path.join('data_' + audio_type))

# Number of MFCC coefficients to consider.
n_mfcc = 13

# Global features need not be aggregated for an audio sample.
global_features = []
# Instantaneous features need to be aggregated for an audio sample.
instantaneous_features = ['rms', 'zcr', 'sc', 'sb', 'sr']
# MFCC features are instantaneous features.
mfcc_features = ['mfcc' + str(i) for i in range(1, n_mfcc + 1)]
instantaneous_features.extend(mfcc_features)

# Functions to aggregate instantaneous features for an audio sample.
# Refer ../eda/feature_extraction_utils.py for allowed values in agg_funcs.
agg_funcs = ['mean', 'median']

# Generating feature names for instantaneous_features x agg_funcs.
instantaneous_features_agg = [instantaneous_feature + '_' + agg_func for instantaneous_feature, agg_func in itertools.product(instantaneous_features, agg_funcs)]

print(f'Using {len(instantaneous_features)} instantaneous features (IF):')
[print(instantaneous_feature, end='\t') for instantaneous_feature in instantaneous_features]
print()

print(f'Using {len(agg_funcs)} aggregation functions (AF):')
[print(agg_func, end='\t') for agg_func in agg_funcs]
print()

print(f'Using {len(global_features)} global features (GF):')
[print(global_feature, end='\t') for global_feature in global_features]
print()

print(f'Total number of features = IF * AF + GF = {len(instantaneous_features_agg) + len(global_features)}')
print()

# All features
features = instantaneous_features_agg + global_features

def generate_feature_row(orig_df, filename, waveform, features, sampling_rate, samples_per_frame, hop_length):
    """ Returns a row of features as a Pandas DataFrame. """

    # row_df will be appended to orig_df, hence must have same columns.
    row_df = pd.DataFrame(columns=orig_df.columns)

    for feature in features:
        # Instantaneous aggregate features contain '_' as substring.
        if '_' in feature:
            feature_name, agg_func = feature.split('_')

            if feature_name == 'rms':
                row_df.loc[filename, feature_name + '_' + agg_func] = rms_energy_agg(waveform, samples_per_frame, hop_length, agg_func)
            elif feature_name == 'zcr':
                row_df.loc[filename, feature_name + '_' + agg_func] = zero_crossing_rate_agg(waveform, samples_per_frame, hop_length, agg_func)
            elif feature_name == 'sc':
                row_df.loc[filename, feature_name + '_' + agg_func] = spectral_centroid_agg(waveform, sampling_rate, samples_per_frame, hop_length, agg_func)
            elif feature_name == 'sb':
                row_df.loc[filename, feature_name + '_' + agg_func] = spectral_bandwidth_agg(waveform, sampling_rate, samples_per_frame, hop_length, agg_func)
            elif feature_name == 'sr':
                row_df.loc[filename, feature_name + '_' + agg_func] = spectral_rolloff_agg(waveform, sampling_rate, samples_per_frame, hop_length, roll_percent, agg_func)
            # Handling MFCC separately.
            elif 'mfcc' in feature_name:
                continue

        # Global features.
        else:
            # No global features yet.
            pass

    # Handling MFCC features separately.
    mfcc_features = [feature for feature in features if 'mfcc' in feature]
    mfcc_max_coef = max([int(mfcc_feature.split('_')[0][4:]) for mfcc_feature in mfcc_features])
    mfcc_agg_funcs = set([mfcc_feature.split('_')[1] for mfcc_feature in mfcc_features])

    for agg_func in agg_funcs:
        # Vector of mfcc_max_coef number of MFCCs.
        mfcc_vec = mfcc_agg(waveform, sampling_rate, samples_per_frame, hop_length, n_mfcc, agg_func)
        for i in range(1, mfcc_max_coef + 1):
            row_df.loc[filename, 'mfcc' + str(i) + '_' + agg_func] = mfcc_vec[i - 1]

    return row_df

for audio_type in audio_types:
    # Paths to train, valid and test directories for each audio type.
    source_train = os.path.join('..', 'data_raw', 'data_' + audio_type, 'train')
    source_valid = os.path.join('..', 'data_raw', 'data_' + audio_type, 'valid')
    source_test = os.path.join('..', 'data_raw', 'data_' + audio_type, 'test')
    # Paths to destination files.
    dest_train = os.path.join('data_' + audio_type, 'train.csv')
    dest_valid = os.path.join('data_' + audio_type, 'valid.csv')
    dest_test = os.path.join('data_' + audio_type, 'test.csv')

    # Train, validation and test dataframes containing handcrafted features.
    train_df = pd.DataFrame(columns=features + ['filename'])
    train_df = train_df.set_index('filename')
    valid_df = pd.DataFrame(columns=features + ['filename'])
    valid_df = valid_df.set_index('filename')
    test_df = pd.DataFrame(columns=features + ['filename'])
    test_df = test_df.set_index('filename')

    print(f'Generating train.csv for audio_type={audio_type}...')
    for dir in sorted(os.listdir(source_train)):
        dir = os.path.join(source_train, dir)
        class_ = os.path.basename(dir)
        for file in tqdm(sorted(os.listdir(dir)), desc=f'class={class_}'):
            file = os.path.join(dir, file)
            waveform, _ = librosa.load(file)

            # Row of features corresponding to 1 audio sample.
            row_df = generate_feature_row(train_df, os.path.basename(file), waveform, features, sampling_rate, samples_per_frame, hop_length)

            # Appending new row to dataframe.
            train_df = train_df.append(row_df)

    # Saving dataframe to disk.
    train_df.to_csv(dest_train)

    print(f'Generating valid.csv for audio_type={audio_type}...')
    for dir in sorted(os.listdir(source_valid)):
        dir = os.path.join(source_valid, dir)
        class_ = os.path.basename(dir)
        for file in tqdm(sorted(os.listdir(dir)), desc=f'class={class_}'):
            file = os.path.join(dir, file)
            waveform, _ = librosa.load(file)

            # Row of features corresponding to 1 audio sample.
            row_df = generate_feature_row(train_df, os.path.basename(file), waveform, features, sampling_rate, samples_per_frame, hop_length)

            # Appending new row to dataframe.
            valid_df = valid_df.append(row_df)

    # Saving dataframe to disk.
    valid_df.to_csv(dest_valid)

    print(f'Generating test.csv for audio_type={audio_type}...')
    for dir in sorted(os.listdir(source_test)):
        dir = os.path.join(source_test, dir)
        class_ = os.path.basename(dir)
        for file in tqdm(sorted(os.listdir(dir)), desc=f'class={class_}'):
            file = os.path.join(dir, file)
            waveform, _ = librosa.load(file)

            # Row of features corresponding to 1 audio sample.
            row_df = generate_feature_row(train_df, os.path.basename(file), waveform, features, sampling_rate, samples_per_frame, hop_length)

            # Appending new row to dataframe.
            test_df = test_df.append(row_df)

    # Saving dataframe to disk.
    test_df.to_csv(dest_test)
