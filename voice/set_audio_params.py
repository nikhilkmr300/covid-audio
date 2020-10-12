# Audio parameters that will be used across all the techniques. Note that you
# will have to run make all in this directory before the changes take effect
# across the board.

import numpy as np
import librosa

# Common parameters
sampling_rate = 16000
samples_per_frame = 256
hop_length = samples_per_frame // 4

# Spectrogram parameters
n_mels = 64

# Feature specific parameters
roll_percent = 0.85     # Percentage for spectral rolloff.

def print_params():
    print('Parameters for audio (you can change these in set_audio_params.py):')
    print(f'\tsampling_rate={sampling_rate}Hz')
    print(f'\tsamples_per_frame={samples_per_frame}')
    print(f'\thop_length={hop_length}')

if __name__ == '__main__':
    print_params()
