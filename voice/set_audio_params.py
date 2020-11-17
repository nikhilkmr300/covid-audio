# Audio parameters that will be used across all the techniques. Note that you
# will have to run make all in this directory before the changes take effect
# across the board.

import numpy as np
import librosa

# Set to True to ignore class asthma for data_rnn, data_spec and data_struc.
# Note that however, for data_raw, data_clean and eda, class asthma samples
# will still be used.
ignore_asthma = True

# Common parameters across ALL models
sampling_rate = 16000
samples_per_frame = 256
hop_length = samples_per_frame // 4
# Looping/clipping breath audio samples to the same length time_per_sample (in
# seconds). Refer eda/eda_audio_len.ipynb for stats on audio sample times.
time_per_sample_breath = 24.29  # 95th percentile value for breath.
time_per_sample_cough = 9.92    # 95th percentile value for cough.

# Parameters for SPECTROGRAM models (CNNs) -- data_spec directory
n_mels = 64

# Parameters for TRADITIONAL ML models -- data_struc directory
struc_global_features = []            # Global features need not be aggregated for an audio sample.
struc_instantaneous_features = ['rmse',
                                'zcr',
                                'sc',
                                'sb',
                                'sr',
                                'mfcc']   # Instantaneous features need to be aggregated for an audio sample.
struc_agg_funcs = [             'mean',
                                'median',
                                'rms',
                                'max',
                                'min',
                                'q1',
                                'q3',
                                # '10',
                                # '90',
                                # 'iqr',
                                'std',
                                # 'skew',
                                # 'kurtosis',
                                'rewm']    # Aggregation functions to use. Refer to data_struc/feature_extraction_utils.py for allowed aggregate functions.
struc_roll_percent = 0.85       # Percentage for spectral rolloff.
struc_n_mfcc = 13               # Number of MFCC coefficients to consider.

# Parameters for RECURRENT models -- data_rnn directory
rnn_instantaneous_features = ['rmse', 'zcr', 'sc', 'sb', 'sr', 'mfcc']   # Using only instantaneous features, without aggregation to preserve time component for RNN. Global features not used.
rnn_roll_percent = 0.85         # Percentage for spectral rolloff.
rnn_n_mfcc = 13                 # Number of MFCC coefficients to consider.

def print_params():
    print('Parameters for audio (you can change these in set_audio_params.py):')
    print(f'\tsampling_rate={sampling_rate}Hz')
    print(f'\tsamples_per_frame={samples_per_frame}')
    print(f'\thop_length={hop_length}')

if __name__ == '__main__':
    print_params()
