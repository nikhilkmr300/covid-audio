import numpy as np
import librosa

from set_audio_params import *

def load(audio_path, sampling_rate, time_per_sample):
    """
    Wrapper around librosa.load. If the audio sample is shorter than
    time_per_sample (you can set this in set_audio_params.py), it is looped back
    to time_per_sample seconds. Else if it is shorter than time_per_sample, it
    is clipped to time_per_sample seconds. Use this in place of librosa.load
    throughout this project.

    Parameters:
    audio_path (str): Absolute/relative path to audio file.
    sampling_rate (float): Number of samples to take per second (discretizing
        time).
    time_per_sample (float): Read description above for explanation.

    Returns:
    tuple: Tuple containing waveform (NumPy array), sampling_rate (float).
    """

    # Standard sampling rate is 44100 Hz.
    waveform, sampling_rate = librosa.load(audio_path, sr=sampling_rate)

    # Looping back if longer, clipping if shorter.
    waveform = np.resize(waveform, int(time_per_sample * sampling_rate))

    return waveform, sampling_rate

if __name__ == '__main__':
    """ Checking if load works fine. """

    import os
    import matplotlib.pyplot as plt
    import librosa.display

    np.random.seed(1)

    # Only for this script. To change the actual value of time_per_sample, set
    # it in set_audio_params.py.
    time_per_sample_local = 100

    # Path to some random audio file.
    choice_dir = os.path.join('data_raw', 'data_clean', 'breath', 'normal')
    choice_file_basenames = os.listdir(choice_dir)
    choice_files = [os.path.join(choice_dir, basename) for basename in choice_file_basenames]
    audio_path = np.random.choice(choice_files)

    print(audio_path)

    audio_custom, _ = load(audio_path, sampling_rate, time_per_sample=time_per_sample_local)
    timesteps_custom = np.linspace(0, len(audio_custom) / sampling_rate, len(audio_custom))
    audio_librosa, _ = librosa.load(audio_path, sr=sampling_rate)
    timesteps_librosa = np.linspace(0, len(audio_librosa) / sampling_rate, len(audio_librosa))

    plt.subplots(2, 1, sharex=True)
    plt.subplots_adjust(hspace=0)

    plt.subplot(2, 1, 1)
    plt.plot(timesteps_custom, audio_custom, color='b', label='Using custom load')
    plt.xlim([0, time_per_sample_local])
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(timesteps_librosa, audio_librosa, color='r', label='Using librosa.load')
    plt.xlim([0, time_per_sample_local])
    plt.xlabel('Time (s)')
    plt.legend()
    plt.show()
