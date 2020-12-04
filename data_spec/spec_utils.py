# Utilities for handling audio samples and spectrograms.

import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt
import librosa
import os

import sys
sys.path.append('..')

from set_audio_params import *
from load_utils import load

def generate_waveform(audio_path, sampling_rate, time_per_sample):
    """
    Returns the audio sample as a NumPy array, along with the sampling rate.
    Uses function load from load_utils.py.

    Parameters:
    audio_path (str): Path to audio file.
    sampling_rate (float): Number of samples to take per second (discretizing
        time).

    Returns:
    tuple: Tuple of waveform as NumPy array and timesteps as NumPy array.
    """

    waveform, _ = load(audio_path, sampling_rate, time_per_sample)

    # Converting from sample index to time in seconds, using the sampling rate.
    timesteps = np.linspace(0, len(waveform) / sampling_rate, len(waveform))

    return waveform, timesteps

def plot_waveform(timesteps, waveform, show=True, figsize=(10, 4), xlabel='Time (s)', ylabel='y', title=None, color='b'):
    """ Plots the waveform of the audio sample. """

    fig = plt.figure(figsize=figsize)
    plt.plot(timesteps, waveform, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if show == True:
        plt.show()

    return fig

def save_waveform(timesteps, waveform, output_path, figsize=(10, 4), xlabel='Time (s)', ylabel='y', title=None, color='b'):
    """ Saves plot of waveform to disk. """

    fig = plot_waveform(timesteps, waveform, show=False, figsize=figsize, xlabel=xlabel, ylabel=ylabel, title=title, color=color)
    fig.savefig(output_path, bbox_inches='tight')

def generate_spectrum(waveform, sampling_rate):
    """ Generates magnitudes and frequencies from FFT, over whole time of audio sample. """

    # Generating the Fourier transform of the waveform, and taking the
    # magnitudes corresponding to each frequency.
    waveform_freq_space = np.abs(np.fft.fft(waveform))
    # Frequencies have range [0, sampling_rate), and discretized to have
    # len(waveform_freq_space) number of values in that range.
    freqs = np.linspace(0, sampling_rate, len(waveform_freq_space))

    return waveform_freq_space, freqs

def plot_spectrum(freqs, waveform_freq_space, show=True, figsize=(10, 4), xlabel='Frequency (Hz)', ylabel='Magnitude', title=None, color='b'):
    """ Plots magnitude vs frequency values returned from function generate_spectrum. """

    fig = plt.figure(figsize=figsize)

    # Power spectrum plots are symmetric around the Nyquist frequency. Plotting
    # for only the lower half of frequencies as no information is lost.
    half_freq_index = int(0.5 * len(freqs))
    freqs = freqs[:half_freq_index]
    waveform_freq_space = waveform_freq_space[:half_freq_index]

    fig = plt.plot(freqs, waveform_freq_space, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if show == True:
        plt.show()

    return fig

def save_spectrum(freqs, waveform_freq_space, output_path, figsize=(10, 4), xlabel='Frequency (Hz)', ylabel='Magnitude', title=None, color='b'):
    """ Saves spectrum to disk. """

    fig = plot_spectrum(freqs, waveform_freq_space, show=False, figsize=figsize, xlabel=xlabel, ylabel=ylabel, title=title, color=color)
    fig.savefig(output_path, bbox_inches='tight')

def generate_spectrogram(waveform, sampling_rate, samples_per_frame, hop_length, window='hann'):
    """
    Generates magnitude as a function of frequency and time, by partitioning time into slices.

    Parameters:
    waveform (NumPy array): Waveform as returned by generate_waveform.
    sampling_rate (float): Same as for generate_waveform.
    samples_per_frame (int): Number of samples (window) over which to take an
        FFT.
    hop_length (int): Number of samples to jump by after taking an FFT.
    window (str): Windowing function as accepted by librosa.stft.

    Returns:
    NumPy array: Spectrogram (2D array)
    """

    # Generating the short time Fourier transform of the waveform, and taking
    # the magnitudes corresponding to each frequency.
    spec = np.abs(librosa.stft(waveform, n_fft=samples_per_frame, hop_length=hop_length, window=window))

    return spec

def save_spectrogram(spec, hop_length, output_path, x_axis='time', y_axis='log', figsize=(10, 4), title=None):
    """ Saves spectrogram to disk. """

    fig = plt.figure()
    plt.style.use('dark_background')
    plt.suptitle(title)
    librosa.display.specshow(spec, hop_length=hop_length, x_axis=x_axis, y_axis=y_axis)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    fig.savefig(output_path)
    fig.clear()
    plt.close('all')
