import matplotlib.pyplot as plt
import librosa
from madmom.features.beats import *
from scipy import signal
import numpy as np

# Beat detect from https://github.com/dodiku/AudioOwl/blob/master/AudioOwl/analyze.py
# Need to install C++ VS toolset and madmom
def peak_picking(beat_times, total_samples, kernel_size, offset):

    # smoothing the beat function
    cut_off_norm = len(beat_times)/total_samples*100/2
    b, a = signal.butter(1, cut_off_norm)
    beat_times = signal.filtfilt(b, a, beat_times)

    # creating a list of samples for the rnn beats
    beat_samples = np.linspace(0, total_samples, len(beat_times), endpoint=True, dtype=int)

    n_t_medians = signal.medfilt(beat_times, kernel_size=kernel_size)
    offset = 0.01
    peaks = []

    for i in range(len(beat_times)-1):
        if beat_times[i] > 0:
            if beat_times[i] > beat_times[i-1]:
                if beat_times[i] > beat_times[i+1]:
                    if beat_times[i] > (n_t_medians[i] + offset):
                        peaks.append(int(beat_samples[i]))
    return peaks


waveform, sr = librosa.load('test.mp3', sr=22050)


plt.figure()
# plt.vlines(data['beat_samples'], -1.0, 1.0)
plt.plot(waveform)
plt.show()