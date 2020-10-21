# Utilities for handling instantaneous, global and aggregate instantaneous
# audio features generated from the audio files in data_clean. Features are
# extracted after the audio samples are changed to the same length, i.e.,
# time_per_sample, refer to set_audio_params.py.

import numpy as np
import scipy.stats
import librosa

# Aggregate functions taken in the KDD paper.
agg_funcs_allowed = [
    'mean',     # Arithmetic mean
    'median',   # Median
    'rms',      # Root mean square value
    'max',      # Maximum
    'min',      # Minimum
    'q1',       # 1st quartile
    'q3',       # 3rd quartile
    'iqr',      # Interquartile range
    'std',      # Standard deviation
    'skew',     # Skewness
    'kurtosis', # Kurtosis
    'rewm'      # A custom aggregation function rms energy weighted mean, not
                # given in the KDD paper.
    # Integer values in the range [0, 100] are also allowed, representing the
    # percentile value in arr. For example, passing 95 would return the 95th
    # percentile value in arr. This too is not used in the KDD paper.
]

# Function to aggregate frame-level/instantaneous features to 1 value for the
# whole audio sample.
def aggregate(arr, agg_func, rms=None):
    if not (agg_func in agg_funcs_allowed or (agg_func.isnumeric() and (0 <= float(agg_func) <= 100))):
        raise ValueError(f'agg_func must be one among {agg_funcs_allowed} or a float in the range [0, 100] represented as a string.')
    if arr.ndim != 1 and arr.ndim != 2:
        raise ValueError(f'arr must be a tensor of rank 1.')

    if agg_func == 'mean':
        # For MFCCs, calculating across time, axis=1.
        if arr.ndim == 2:
            return np.mean(arr, axis=1)
        return np.mean(arr)
    elif agg_func == 'median':
        # For MFCCs, calculating across time, axis=1.
        if arr.ndim == 2:
            return np.median(arr, axis=1)
        return np.median(arr)
    elif agg_func == 'rms':
        # For MFCCs, calculating across time, axis=1.
        if arr.ndim == 2:
            return np.sqrt(np.sum(arr ** 2, axis=1) / arr.shape[1])
        return np.sqrt(np.sum(arr ** 2) / len(arr))
    elif agg_func == 'max':
        # For MFCCs, calculating across time, axis=1.
        if arr.ndim == 2:
            return np.max(arr, axis=1)
        return np.max(arr)
    elif agg_func == 'min':
        # For MFCCs, calculating across time, axis=1.
        if arr.ndim == 2:
            return np.min(arr, axis=1)
        return np.min(arr)
    elif agg_func == 'q1':
        # For MFCCs, calculating across time, axis=1.
        if arr.ndim == 2:
            return np.percentile(arr, 25, axis=1)
        return np.percentile(arr, 25)
    elif agg_func == 'q3':
        # For MFCCs, calculating across time, axis=1.
        if arr.ndim == 2:
            return np.percentile(arr, 75, axis=1)
        return np.percentile(arr, 75)
    elif agg_func == 'iqr':
        # For MFCCs, calculating across time, axis=1.
        if arr.ndim == 2:
            return np.percentile(arr, 75, axis=1) - np.percentile(arr, 25, axis=1)
        return np.percentile(arr, 75) - np.percentile(arr, 25)
    elif agg_func == 'std':
        # For MFCCs, calculating across time, axis=1.
        if arr.ndim == 2:
            return np.std(arr, axis=1)
        return np.std(arr)
    elif agg_func == 'skew':
        # For MFCCs, calculating across time, axis=1.
        if arr.ndim == 2:
            return scipy.stats.skew(arr, axis=1)
        return scipy.stats.skew(arr)
    elif agg_func == 'kurtosis':
        # For MFCCs, calculating across time, axis=1.
        if arr.ndim == 2:
            return scipy.stats.kurtosis(arr, axis=1)
        return scipy.stats.kurtosis(arr)
    elif agg_func == 'rewm':
        # Using this option requires RMS energy vector.
        if rms is None:
            raise ValueError('aggregate with agg_func as rms_energy_weighted_mean requires rms parameter.')
        # Handles case of MFCC matrix as well, which has shape (struc_n_mfcc, num_frames).
        return np.dot(arr, rms) / np.sum(rms)
    elif agg_func.isnumeric() and 0 <= float(agg_func) <= 100:
        # For MFCCs, calculating across time, axis=1.
        if arr.ndim == 2:
            return np.percentile(arr, float(agg_func), axis=1)
        return np.percentile(arr, float(agg_func))

# INSTANTANEOUS FEATURES
# Wrappers around librosa functions that:
# 1. Use more intuitive names.
# 2. Convert optional arguments to compulsory arguments. I've spent too much
#    time debugging before just to realize later that I hadn't provided an
#    optional argument that was required to generate a desired result.
# 3. Get rid of distracting options not required by this project.
def rms_energy(waveform, samples_per_frame, hop_length):
    return librosa.feature.rms(y=waveform, frame_length=samples_per_frame, hop_length=hop_length).flatten()

def zero_crossing_rate(waveform, samples_per_frame, hop_length):
    return librosa.feature.zero_crossing_rate(waveform, frame_length=samples_per_frame, hop_length=hop_length).flatten()

def spectral_centroid(waveform, sampling_rate, samples_per_frame, hop_length):
    return librosa.feature.spectral_centroid(waveform, sr=sampling_rate, n_fft=samples_per_frame, hop_length=hop_length).flatten()

def spectral_bandwidth(waveform, sampling_rate, samples_per_frame, hop_length):
    return librosa.feature.spectral_bandwidth(waveform, sr=sampling_rate, n_fft=samples_per_frame, hop_length=hop_length).flatten()

def spectral_rolloff(waveform, sampling_rate, samples_per_frame, hop_length, roll_percent):
    return librosa.feature.spectral_rolloff(waveform, sr=sampling_rate, n_fft=samples_per_frame, hop_length=hop_length, roll_percent=roll_percent).flatten()

def mfcc(waveform, sampling_rate, samples_per_frame, hop_length, n_mfcc):
    return librosa.feature.mfcc(waveform, sr=sampling_rate, n_fft=samples_per_frame, hop_length=hop_length, n_mfcc=n_mfcc)

def dmfcc(waveform, sampling_rate, samples_per_frame, hop_length, n_mfcc):
    mfcc = librosa.feature.mfcc(waveform, sr=sampling_rate, n_fft=samples_per_frame, hop_length=hop_length, n_mfcc=n_mfcc)
    return librosa.feature.delta(mfcc)

def d2mfcc(waveform, sampling_rate, samples_per_frame, hop_length, n_mfcc):
    mfcc = librosa.feature.mfcc(waveform, sr=sampling_rate, n_fft=samples_per_frame, hop_length=hop_length, n_mfcc=n_mfcc)
    return librosa.feature.delta(mfcc, order=2)

# AGGREGATE INSTANTANEOUS FEATURES
# Note that aggregate function 'rewm' requires slightly different treatment (it
# requires the root mean square energies rms to be passed to the aggregate
# function), because of the definition of 'rewm'.
def rms_energy_agg(waveform, samples_per_frame, hop_length, agg_func='95', rms=None):
    """ Returns aggregate of framewise RMS energies, for an audio clip. """
    rms_energies = rms_energy(waveform, samples_per_frame, hop_length)
    if agg_func == 'rewm':
        # Using RMS energy to weight frames. Frames with higher RMS energy
        # contribute more to aggregate rms energy.
        # I don't know if it makes sense to weight rms energy with rms energy
        # to aggregate it, it'd just be squaring the rms energies over the
        # frames, and taking their mean. Keeping it for the sake of consistency.
        # If required, it can be removed from the csv files.
        return aggregate(rms_energies, agg_func, rms=rms)
    return aggregate(rms_energies, agg_func)

def zero_crossing_rate_agg(waveform, samples_per_frame, hop_length, agg_func, rms=None):
    """ Returns aggregate of framewise zero crossing rates, for an audio clip. """

    zcr = zero_crossing_rate(waveform, samples_per_frame, hop_length)
    if agg_func == 'rewm':
        # Using RMS energy to weight frames. Frames with higher RMS energy
        # contribute more to aggregate zero crossing rate.
        rms = rms_energy(waveform, samples_per_frame, hop_length)
        return aggregate(zcr, agg_func, rms=rms)
    return aggregate(zcr, agg_func)

def spectral_centroid_agg(waveform, sampling_rate, samples_per_frame, hop_length, agg_func, rms=None):
    """ Returns aggregate of spectral centroids, for an audio clip. """

    spec_centroids = spectral_centroid(waveform, sampling_rate, samples_per_frame, hop_length)
    if agg_func == 'rewm':
        # Using RMS energy to weight frames. Frames with higher RMS energy
        # contribute more to aggregate zero crossing rate.
        rms = rms_energy(waveform, samples_per_frame, hop_length)
        return aggregate(spec_centroids, agg_func, rms=rms)
    return aggregate(spec_centroids, agg_func)

def spectral_bandwidth_agg(waveform, sampling_rate, samples_per_frame, hop_length, agg_func, rms=None):
    """ Returns aggregate of framewise spectral bandwidths, for an audio clip. """

    spec_bws = spectral_bandwidth(waveform, sampling_rate, samples_per_frame, hop_length)
    if agg_func == 'rewm':
        # Using RMS energy to weight frames. Frames with higher RMS energy
        # contribute more to aggregate zero crossing rate.
        rms = rms_energy(waveform, samples_per_frame, hop_length)
        return aggregate(spec_bws, agg_func, rms=rms)
    return aggregate(spec_bws, agg_func)

def spectral_rolloff_agg(waveform, sampling_rate, samples_per_frame, hop_length, roll_percent, agg_func, rms=None):
    """ Returns aggregate of framewise spectral rolloffs, for an audio clip. """

    spec_rolloffs = spectral_rolloff(waveform, sampling_rate, samples_per_frame, hop_length, roll_percent)
    if agg_func == 'rewm':
        # Using RMS energy to weight frames. Frames with higher RMS energy
        # contribute more to aggregate zero crossing rate.
        rms = rms_energy(waveform, samples_per_frame, hop_length)
        return aggregate(spec_rolloffs, agg_func, rms=rms)
    return aggregate(spec_rolloffs, agg_func)

def mfcc_agg(waveform, sampling_rate, samples_per_frame, hop_length, n_mfcc, agg_func, rms=None):
    """ Returns aggregate across time axis (axis=1) of MFCCs, for an audio clip. """

    mfccs = mfcc(waveform, sampling_rate, samples_per_frame, hop_length, n_mfcc)
    if agg_func == 'rewm':
        # Using RMS energy to weight frames. Frames with higher RMS energy
        # contribute more to aggregate zero crossing rate.
        rms = rms_energy(waveform, samples_per_frame, hop_length)
        return aggregate(mfccs, agg_func, rms=rms)
    return aggregate(mfccs, agg_func)

def dmfcc_agg(waveform, sampling_rate, samples_per_frame, hop_length, n_mfcc, agg_func, rms=None):
    """ Returns aggregate across time axis (axis=1) of derivative of MFCCs, for an audio clip. """

    dmfccs = dmfcc(waveform, sampling_rate, samples_per_frame, hop_length, n_mfcc)
    if agg_func == 'rewm':
        # Using RMS energy to weight frames. Frames with higher RMS energy
        # contribute more to aggregate zero crossing rate.
        rms = rms_energy(waveform, samples_per_frame, hop_length)
        return aggregate(dmfccs, agg_func, rms=rms)
    return aggregate(dmfccs, agg_func)

def d2mfcc_agg(waveform, sampling_rate, samples_per_frame, hop_length, n_mfcc, agg_func, rms=None):
    """ Returns aggregate across time axis (axis=1) of second derivative of MFCCs, for an audio clip. """

    d2mfccs = d2mfcc(waveform, sampling_rate, samples_per_frame, hop_length, n_mfcc)
    if agg_func == 'rewm':
        # Using RMS energy to weight frames. Frames with higher RMS energy
        # contribute more to aggregate zero crossing rate.
        rms = rms_energy(waveform, samples_per_frame, hop_length)
        return aggregate(d2mfccs, agg_func, rms=rms)
    return aggregate(d2mfccs, agg_func)
