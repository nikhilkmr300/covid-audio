# Script to generate train, valid and test csv files for traditional ML models.

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
sys.path.append('..')

from set_audio_params import *
from load_utils import *
from feature_extraction_utils import *

def generate_feature_names():
    """
    Generates names of feature columns to be used in the training and test dataframes from the feature names mentioned in set_audio_params.py.

    Parameters:
    None

    Returns:
    list: List of names of feature columns to be used in the training and test dataframes.
    """

    # MFCC features are instantaneous features.
    if 'mfcc' in struc_instantaneous_features:
        mfcc_features = ['mfcc' + str(i) for i in range(1, struc_n_mfcc + 1)]
        struc_instantaneous_features.extend(mfcc_features)

    # Removing the dummy literal 'mfcc' which stood for all coefficients from mfcc0
    # to mfcc<struc_n_mfcc>, as we have already handled the mfcc features above.
    struc_instantaneous_features.remove('mfcc')

    # Generating feature names for struc_instantaneous_features x struc_agg_funcs.
    struc_instantaneous_features_agg = [instantaneous_feature + '_' + str(agg_func) for instantaneous_feature, agg_func in itertools.product(struc_instantaneous_features, struc_agg_funcs)]

    print(f'Using {len(struc_instantaneous_features)} instantaneous features (IF):')
    [print(instantaneous_feature, end='\t') for instantaneous_feature in struc_instantaneous_features]
    print()

    print(f'Using {len(struc_agg_funcs)} aggregation functions (AF):')
    [print(agg_func, end='\t') for agg_func in struc_agg_funcs]
    print()

    print(f'Using {len(struc_global_features)} global features (GF):')
    [print(global_feature, end='\t') for global_feature in struc_global_features]
    print()

    print(f'Total number of features = IF * AF + GF = {len(struc_instantaneous_features_agg) + len(struc_global_features)}')
    print()

    # All features
    features = struc_instantaneous_features_agg + struc_global_features

    return features

def generate_feature_row(orig_df, filename, class_, waveform, sampling_rate, samples_per_frame, hop_length):
    """ Returns a row of features as a Pandas DataFrame. """

    # row_df will be appended to orig_df, hence must have same columns.
    row_df = pd.DataFrame(columns=orig_df.columns)

    # Pre-calculating rms energy if agg_func is rewm. Remember that rewm
    # requires slightly different treatment than the other aggregate functions.
    # Passing it as a parameter regardless of the aggregate functions, if it is
    # not rewm, rms is ignored.
    rms = rms_energy(waveform, samples_per_frame, hop_length)

    for feature in row_df.columns:
        # Instantaneous aggregate features contain '_' as substring.
        if '_' in feature:
            feature_name, agg_func = feature.split('_')

            if feature_name == 'rmse':
                row_df.loc[filename, feature_name + '_' + agg_func] = rms_energy_agg(waveform, samples_per_frame, hop_length, agg_func, rms=rms)
            elif feature_name == 'zcr':
                row_df.loc[filename, feature_name + '_' + agg_func] = zero_crossing_rate_agg(waveform, samples_per_frame, hop_length, agg_func, rms=rms)
            elif feature_name == 'sc':
                row_df.loc[filename, feature_name + '_' + agg_func] = spectral_centroid_agg(waveform, sampling_rate, samples_per_frame, hop_length, agg_func, rms=rms)
            elif feature_name == 'sb':
                row_df.loc[filename, feature_name + '_' + agg_func] = spectral_bandwidth_agg(waveform, sampling_rate, samples_per_frame, hop_length, agg_func, rms=rms)
            elif feature_name == 'sr':
                row_df.loc[filename, feature_name + '_' + agg_func] = spectral_rolloff_agg(waveform, sampling_rate, samples_per_frame, hop_length, struc_roll_percent, agg_func, rms=rms)
            # Handling MFCC separately.
            elif 'mfcc' in feature_name:
                continue

        elif feature == 'target':
            row_df.loc[filename, feature] = class_

        # Global features.
        else:
            # No global features yet.
            pass

    # Handling MFCC features separately.
    mfcc_features = [feature for feature in orig_df.columns.tolist() if 'mfcc' in feature]
    # Sanity check that max mfcc coefficient in mfcc_features is same as struc_n_mfcc from set_audio_params.py.
    assert struc_n_mfcc == max([int(mfcc_feature.split('_')[0][4:]) for mfcc_feature in mfcc_features])
    mfcc_struc_agg_funcs = set([mfcc_feature.split('_')[1] for mfcc_feature in mfcc_features])

    for agg_func in struc_agg_funcs:
        # Vector of mfcc_max_coef number of MFCCs.
        mfcc_vec = mfcc_agg(waveform, sampling_rate, samples_per_frame, hop_length, struc_n_mfcc, agg_func, rms=rms)
        for i in range(1, struc_n_mfcc + 1):
            row_df.loc[filename, 'mfcc' + str(i) + '_' + str(agg_func)] = mfcc_vec[i - 1]

    return row_df

def add_is_cough_symptom(filename):
    """ Returns whether cough is a symptom, using the filename (index of the dataframe). """

    # Filename convention comes useful here.
    # Filename convention comes useful here.
    is_symptom = filename.split('_')[2]

    if 'with' in is_symptom:
        return 1
    elif 'no' in is_symptom:
        return 0
    else:
        print('Make sure the filename convention is followed.')
        sys.exit(1)

# Not used. Adding target handled in generate_feature_row.
def add_target(filename):
    """ Returns the target class, using the filename (index of the dataframe). """

    target = filename.split('_')[1].lower()

    if target == 'covid':
        return 1
    elif target == 'asthma' or target == 'normal':
        return 0
    else:
        print('Make sure the filename convention is followed.')
        sys.exit(1)

if __name__ == '__main__':
    # Types of audio (currently breath and cough).
    audio_types = os.listdir(os.path.join('..', 'data_raw', 'data_clean'))

    # Making directories for each audio type.
    for audio_type in audio_types:
        if os.path.exists('data_' + audio_type):
            shutil.rmtree('data_' + audio_type)
        os.makedirs(os.path.join('data_' + audio_type))

    print_params()
    print()

    features = generate_feature_names()

    for audio_type in sorted(audio_types):
        # Paths to train, valid and test directories for each audio type.
        source_train = os.path.join('..', 'data_raw', 'data_' + audio_type, 'train')
        source_valid = os.path.join('..', 'data_raw', 'data_' + audio_type, 'valid')
        source_test = os.path.join('..', 'data_raw', 'data_' + audio_type, 'test')
        # Paths to destination files.
        dest_train = os.path.join('data_' + audio_type, 'train.csv')
        dest_valid = os.path.join('data_' + audio_type, 'valid.csv')
        dest_test = os.path.join('data_' + audio_type, 'test.csv')

        # Train, validation and test dataframes containing handcrafted features.
        train_df = pd.DataFrame(columns=features + ['filename', 'target'])
        train_df = train_df.set_index('filename')
        valid_df = pd.DataFrame(columns=features + ['filename', 'target'])
        valid_df = valid_df.set_index('filename')
        test_df = pd.DataFrame(columns=features + ['filename', 'target'])
        test_df = test_df.set_index('filename')

        print(f'Generating train.csv for audio_type={audio_type}...')
        for dir in sorted(os.listdir(source_train)):
            dir = os.path.join(source_train, dir)
            class_ = os.path.basename(dir)

            # Ignoring class asthma.
            if class_ == 'asthma':
                print('Ignoring class=asthma.')
                continue

            for file in tqdm(sorted(os.listdir(dir)), desc=f'class={class_}'):
                file = os.path.join(dir, file)

                if audio_type == 'breath':
                    waveform, _ = load(file, sampling_rate, time_per_sample_breath)
                elif audio_type == 'cough':
                    waveform, _ = load(file, sampling_rate, time_per_sample_cough)

                # Row of features corresponding to 1 audio sample.
                row_df = generate_feature_row(train_df, os.path.basename(file), class_, waveform, sampling_rate, samples_per_frame, hop_length)

                # Appending new row to dataframe.
                train_df = train_df.append(row_df)

        # Adding is_cough_symptom and target values using filename.
        train_df['is_cough_symptom'] = train_df.index.to_series().apply(add_is_cough_symptom)

        # Saving dataframe to disk.
        train_df.to_csv(dest_train)

        print(f'Generating valid.csv for audio_type={audio_type}...')
        for dir in sorted(os.listdir(source_valid)):
            dir = os.path.join(source_valid, dir)
            class_ = os.path.basename(dir)

            # Ignoring class asthma.
            if class_ == 'asthma':
                print('Ignoring class=asthma.')
                continue

            for file in tqdm(sorted(os.listdir(dir)), desc=f'class={class_}'):
                file = os.path.join(dir, file)

                if audio_type == 'breath':
                    waveform, _ = load(file, sampling_rate, time_per_sample_breath)
                elif audio_type == 'cough':
                    waveform, _ = load(file, sampling_rate, time_per_sample_cough)

                # Row of features corresponding to 1 audio sample.
                row_df = generate_feature_row(valid_df, os.path.basename(file), class_, waveform, sampling_rate, samples_per_frame, hop_length)

                # Appending new row to dataframe.
                valid_df = valid_df.append(row_df)

        # Adding is_cough_symptom and target values using filename.
        valid_df['is_cough_symptom'] = valid_df.index.to_series().apply(add_is_cough_symptom)

        # Saving dataframe to disk.
        valid_df.to_csv(dest_valid)

        print(f'Generating test.csv for audio_type={audio_type}...')
        for dir in sorted(os.listdir(source_test)):
            dir = os.path.join(source_test, dir)
            class_ = os.path.basename(dir)

            # Ignoring class asthma.
            if class_ == 'asthma':
                print('Ignoring class=asthma.')
                continue

            for file in tqdm(sorted(os.listdir(dir)), desc=f'class={class_}'):
                file = os.path.join(dir, file)

                if audio_type == 'breath':
                    waveform, _ = load(file, sampling_rate, time_per_sample_breath)
                elif audio_type == 'cough':
                    waveform, _ = load(file, sampling_rate, time_per_sample_cough)

                # Row of features corresponding to 1 audio sample.
                row_df = generate_feature_row(test_df, os.path.basename(file), class_, waveform, sampling_rate, samples_per_frame, hop_length)

                # Appending new row to dataframe.
                test_df = test_df.append(row_df)

        # Adding is_cough_symptom and target values using filename.
        test_df['is_cough_symptom'] = test_df.index.to_series().apply(add_is_cough_symptom)

        # Saving dataframe to disk.
        test_df.to_csv(dest_test)
