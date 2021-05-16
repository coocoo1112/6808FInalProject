import sounddevice as sd
import numpy as np
from scipy.signal import chirp
import queue
import threading
import sys
import shutil
import math
import datetime
import scipy.io.wavfile as wav
import json
import matplotlib.pyplot as plt



buff_size = 40
block_size = 4800
q = queue.Queue(maxsize=buff_size)
event = threading.Event()

indatas = []
outdatas = []
previous = None
step = 0
first_peaks = []
calibration_steps = 10
window_range = None
argmaxes = []
distances = []


def get_largest_n_mean(array, n):
    return np.mean(np.argpartition(array, -n)[-n:])

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w



def callback(indata, outdata, frames, time, status):
    global previous
    global step
    global first_peaks
    global calibration_steps
    global window_range
    global argmaxes
    try:
        data = q.get_nowait()
        
    except queue.Empty as e:
        print('Buffer is empty: increase buffersize?', file=sys.stderr)
        raise sd.CallbackAbort from e
    if any(indata):
        try:
            multiplied = np.multiply(np.copy(indata), data.reshape((block_size, 1)))
        except:
            raise sd.CallbackStop
        fft = np.fft.rfft(multiplied.reshape((block_size, 1))[:, 0])
        subtracted = np.subtract(fft, previous) if previous is not None else fft
        previous = np.copy(fft)
        print(np.argmax(np.abs(subtracted)))
        # if step <= calibration_steps:
        first_peaks.append(np.argmax(np.abs(subtracted)))
        # elif step == calibration_steps + 1: #may need to increase this
        PEAK_WINDOW_SIZE = 100
        median_peak_location = int(np.median(first_peaks))
        window_range = np.arange(median_peak_location - PEAK_WINDOW_SIZE / 2,
                    median_peak_location + PEAK_WINDOW_SIZE / 2,
                    dtype=np.int32)
        if step > calibration_steps:
            subtracted_filtered = subtracted[window_range]
            argmax = np.argmax(np.abs(subtracted_filtered))
            MEAN_WINDOW = 2
            mean_argmax = get_largest_n_mean(subtracted_filtered, MEAN_WINDOW)
            argmaxes.append(mean_argmax)
            print("MEAN: ", mean_argmax)
            # freqs = np.fft.rfftfreq(len(subtracted_filtered))
            # freq = freqs[int(mean_argmax)]
            # freq_in_hertz = abs(freq * fs)
            # distance = freq_in_hertz * 343 * .1 / 6000
            # distances.append(distance)



        # if previous is None:
        #     previous = np.copy(fft)
        # subtracted = np.subtract(fft, previous)
        # previous = np.copy(fft)
        # peak = np.argmax(subtracted)
        step += 1
    else:
        print('no input')
    if len(data) < len(outdata):
        print(len(data), len(outdata))
        outdata[:len(data)] = data.reshape(len(data), 1)
        outdata[len(data):] = np.zeros(((len(outdata) - len(data), 1)))
        raise sd.CallbackStop
    else:
        outdata[:] = data.reshape((block_size, 1))


print(sd.query_devices())
fs = int(sd.query_devices(0, 'input')['default_samplerate'])
T = .1
t = np.linspace(0, T, int(T*fs), endpoint=False)
w = chirp(t, f0=17000, f1=23000, t1=T, method='linear').astype(np.float32)
scaled = None
for i in range(100):
    if scaled is None:
        scaled = np.array(w)
    else:
        scaled = np.concatenate((scaled,w))

for _ in range(20):
    data = scaled[:min(block_size, len(scaled))]#,0]
    if len(data) == 0:
        break
    scaled = scaled[min(block_size, len(scaled)):]
    q.put_nowait(data)

with sd.Stream(device=(0,1), samplerate=fs, dtype='float32', latency='low', channels=(1,1), callback=callback, blocksize=block_size, finished_callback=event.set):
    timeout = block_size * buff_size / fs
    while len(data) != 0:
        data = scaled[:min(block_size, len(scaled))]#,0]
        scaled = scaled[min(block_size, len(scaled)):]
        q.put(data, timeout=timeout)
    event.wait()
    plt.plot(argmaxes)
    plt.show()
    # plt.plot(distances)
    # plt.show()
    