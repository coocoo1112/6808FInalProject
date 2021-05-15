import numpy as np
import matplotlib.pyplot as plt


np_name = "experiment-up-sweep-1.npy"

arr = np.load(np_name)
print(arr.shape)
indata = arr[0]
outdata = arr[1]
zeros = np.zeros(len(indata[0]))
block_size = 4800
fs = 48000
peak_freqs = []
previous = None

for i, chirp in enumerate(indata):
    multiplied = np.multiply(chirp, outdata[i])
    fft = np.fft.rfft(multiplied.reshape((block_size, 1))[:, 0])
    if previous is None:
        previous = fft
    subtracted_fft = np.subtract(fft, previous)
    previous = np.copy(fft)
    if i%10 == 0:
        freqs = np.fft.rfftfreq(block_size)#, d=1/fs)
        print(freqs)
        x_ticks = []
        for idx in range(len(freqs)):
            
            freq_in_hertz = abs(freqs[idx] * fs)
            x_ticks.append(freq_in_hertz)
        plt.plot(x_ticks, subtracted_fft)
        plt.show()
    idx = np.argmax(np.abs(subtracted_fft))
    freqs = np.fft.rfftfreq(block_size)
    freq = freqs[idx]
    freq_in_hertz = abs(freq * fs)
    peak_freqs.append(freq_in_hertz)

for i in range(len(peak_freqs) - 1):
    print(peak_freqs[i+1] - peak_freqs[i])
print(peak_freqs[-1]-max(peak_freqs[4:]))
plt.plot(peak_freqs)
plt.show()
