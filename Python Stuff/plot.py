#!/usr/bin/env python3

import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

SAMPLING_FREQ = 48000   # iOS samples at 48k by default
SWEEP_T       = 0.02    # length, in seconds, of each sweep

def process_waveform(waveform):
    # get rid of final trailing comma
    waveform = waveform[:-1]

    # trim to exactly SAMPLING_FREQ * 5 samples
    waveform = waveform[:SAMPLING_FREQ * 5]
    print(waveform.shape)

    if waveform.shape[0] < SAMPLING_FREQ * 5:
        print("sample isn't long enough")
        exit(-1)

# from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def filter_waveform(waveform):
    LOW_FREQ = 17000
    HIGH_FREQ = 23000
    return butter_bandpass_filter(waveform, LOW_FREQ, HIGH_FREQ, SAMPLING_FREQ, order=6)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('needs filename')
        exit(-1)
    
    if len(sys.argv) == 2:
        waveform = np.genfromtxt(sys.argv[1], delimiter=',')
        # filter waveform, idk if this does anything
        waveform = filter_waveform(waveform)
        plt.title("Matplotlib demo")
        plt.xlabel("time (s)")
        plt.ylabel("freq (Hz)")
        plt.figure(1)
        # f, t, Sxx = signal.spectrogram(waveform, SAMPLING_FREQ)
        # f, t, Sxx = signal.spectrogram(waveform, SAMPLING_FREQ, nperseg=64, noverlap=0)

        f, t, Sxx = signal.spectrogram(waveform, SAMPLING_FREQ, nperseg=64 * 30, noverlap=0)
        # plt.pcolormesh(t, f, Sxx, shading='gouraud')
        # plt.pcolormesh(t[410:470], f, 10*np.log10(Sxx[:, 410:470]), shading='gouraud')
        plt.pcolormesh(t, f, 10*np.log10(Sxx), shading='gouraud')
        # plt.specgram(waveform, Fs=48000)
        print(Sxx.shape)

        # subtract Sxx from itself, but shifted by SWEEP_T * SAMPLING_FREQ
        # this is the same as subtracting the current sweep's FFT from
        # the previous FFT's sweep (i.e. background subtraction)
        # sweep_width = int(SWEEP_T * SAMPLING_FREQ)
        plt.figure(2)
        # sweep_width = 30
        # subtracted = Sxx[:, sweep_width:] - Sxx[:, :-sweep_width]
        subtracted = Sxx[:, 1:] - Sxx[:, :-1]
        print(subtracted.shape, 1, t.shape)
        plt.pcolormesh(t[:-1], f, 10*np.log10(subtracted), shading='gouraud')

        plt.figure(3)
        max_elements = np.argmax(subtracted, axis=0)
        print("subtracted:")
        print(subtracted.shape)
        print(max_elements.shape)
        plt.plot(f[max_elements])


        # plt.figure(3)
        # plt.pcolormesh(t[:-sweep_width], f, 10*np.log10(subtracted), shading='gouraud')
        # print(Sxx[0:4])
        # print(np.pad(Sxx[0:4]))


        plt.show()

    else:
        waveform1 = np.genfromtxt(sys.argv[1], delimiter=',')
        process_waveform(waveform1)

        waveform2 = np.genfromtxt(sys.argv[2], delimiter=',')
        process_waveform(waveform2)

        f, t, Sxx1 = signal.spectrogram(waveform1, SAMPLING_FREQ)
        _, _, Sxx2 = signal.spectrogram(waveform2, SAMPLING_FREQ)
        # print(f, t, Sxx)

        plt.figure(1)
        # plt.pcolormesh(t, f, Sxx1, shading='gouraud')
        plt.pcolormesh(t, f, 10*np.log10(Sxx1), shading='gouraud')

        plt.figure(2)
        # plt.pcolormesh(t, f, Sxx2, shading='gouraud')
        plt.pcolormesh(t, f, 10*np.log10(Sxx2), shading='gouraud')

        # plt.figure(3)
        # plt.pcolormesh(t, f, Sxx1 - Sxx2, shading='gouraud')

        # find the cross-correlation of the two spectrograms
        correlation = signal.correlate(Sxx1, Sxx2, mode="same")
        plt.figure(3)
        plt.plot(correlation)

        plt.figure(4)

        plt.pcolormesh(t, f, 10*np.log10(Sxx1 - Sxx2), shading='gouraud')
        plt.show()
