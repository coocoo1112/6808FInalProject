import sounddevice as sd
import numpy as np
from scipy.signal import chirp, medfilt, butter, sosfilt, lfilter, windows
import scipy.ndimage as ndimage
import queue
import threading
import sys
import shutil
import math
import datetime
import scipy.io.wavfile as wav
import json
import matplotlib.pyplot as plt
import copy
from kalman_filter import get_position_update
import time

time.sleep(7)

buff_size = 40
block_size = 4800
q = queue.Queue(maxsize=buff_size)
event = threading.Event()

indatas = []
outdatas = []
previous_fmcw = None
step = 0
first_peaks = []
calibration_steps = 10
window_range = None
argmaxes = []
distances = []
argmax_distances = []
keep_going = True

# Doppler variables
doppler_velocities = np.array([])
doppler_distances = np.array([])
doppler_calibration_ffts = []
doppler_calibration_val = None
doppler_velocity = 0
kalman_distances = []

DOPPLER_HANN_WINDOW = windows.hann(block_size)
SAMPLE_RATE = 48000

# Doppler tuning constants
GAUSSIAN_SMOOTHING_SIGMA   = 4      # idk how much of an impact this has, anything from 1-5 seems to work
DOPPLER_VELOCITY_THRESHOLD = 0.5    # higher = less sensitive, but lower noise
DOPPLER_SMOOTHING_FACTOR   = 0.5    # between 0 and 1. higher = more smoothing, slower to react

i=0
f,ax = plt.subplots(4)

x = np.arange(10000)
y = np.random.randn(10000)

# Plot 0 is for raw audio data
li, = ax[0].plot(x, y)
ax[0].set_xlim(0,200)
ax[0].set_ylim(-.5,1)
ax[0].set_title("FMCW Distance Measurements")

li2, = ax[1].plot(x, y)
ax[1].set_xlim(0,200)
ax[1].set_ylim(-5,5)
ax[1].set_title("Doppler Velocity")

li3, = ax[2].plot(x, y)
ax[2].set_xlim(0,200)
ax[2].set_ylim(-20,20)
ax[2].set_title("Doppler Distance")

li4, = ax[3].plot(x, y)
ax[3].set_xlim(0,200)
ax[3].set_ylim(-.5,1)
ax[3].set_title("Kalman Combined Distance")

plt.pause(0.01)
plt.tight_layout()


def get_largest_n_mean(array, n):
    return np.mean(np.argpartition(array, -n)[-n:])

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def is_outlier(arr, new_val):
    q75, q25 = np.percentile(arr, [75 ,25])
    iqr = q75 - q25
    low = q25 - 1.5 * iqr
    high = q75 + 1.5 * iqr
    if new_val < low or new_val > high:
        return True
    return False

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandstop(lowcut, highcut, fs, order=7):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandstop_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandstop(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def get_distance_from_peak(idx, window_range_start, median_peak_location):
    # use global values
    return idx + window_range_start - median_peak_location

def idx_to_distance(idx, freqs):
    global T
    FREQ_PER_FFT_BIN = freqs[1] - freqs[0]  # idk
    SPEED_OF_SOUND = 343
    CHIRP_LENGTH = T
    FREQ_HIGH = 23000
    FREQ_LOW = 17000
    return idx * FREQ_PER_FFT_BIN * SPEED_OF_SOUND * CHIRP_LENGTH / (FREQ_HIGH - FREQ_LOW) #/ 2

# find index in `array` with value closest to `value`
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def update_plot():
    global argmax_distances
    li.set_xdata(np.arange(len(argmax_distances)))
    li.set_ydata(argmax_distances)
    li2.set_xdata(np.arange(len(doppler_velocities)))
    li2.set_ydata(doppler_velocities)
    li3.set_xdata(np.arange(len(doppler_distances)))
    li3.set_ydata(doppler_distances)
    li4.set_xdata(np.arange(len(kalman_distances)))
    li4.set_ydata(np.array(kalman_distances))
    # li2.set_xdata(np.arange(10000))
    # li2.set_ydata(np.random.randn(10000))
    plt.draw()
    plt.pause(0.001)



def callback(indata, outdata, frames, time, status):
    global previous_fmcw
    global step
    global first_peaks
    global calibration_steps
    global window_range
    global argmaxes
    global argmax_distances
    global keep_going
    global kalman_distances
    try:
        data = q.get_nowait()
        
    except queue.Empty as e:
        print('Buffer is empty: increase buffersize?', file=sys.stderr)
        raise sd.CallbackAbort from e
    if any(indata):
        try:
            fmcw_indata = butter_bandstop_filter(indata, 9900, 10100, 48000)
            fmcw_outdata = butter_bandstop_filter(data, 9900, 10100, 48000)
            doppler_indata = butter_bandpass_filter(indata, 9900, 10100, 48000)
            doppler_outdata = butter_bandpass_filter(data, 9900, 10100, 48000)
            multiplied_fmcw = np.multiply(np.copy(fmcw_indata), fmcw_outdata.reshape((block_size, 1)))
        except:
            raise sd.CallbackStop
        fft_fmcw = np.fft.rfft(multiplied_fmcw.reshape((block_size, 1))[:, 0])
        subtracted_fmcw = np.subtract(fft_fmcw, previous_fmcw) if previous_fmcw is not None else fft_fmcw
        previous_fmcw = np.copy(fft_fmcw)
        #print(np.argmax(np.abs(subtracted_fmcw)))
        # if step <= calibration_steps:
        first_peaks.append(np.argmax(np.abs(subtracted_fmcw)))
        # elif step == calibration_steps + 1: #may need to increase this
        PEAK_WINDOW_SIZE = 50
        median_peak_location = int(np.median(first_peaks))
        window_range_start = median_peak_location
        window_range = np.arange(window_range_start,        
                         window_range_start + PEAK_WINDOW_SIZE,
                         dtype=np.int32)

        # Doppler calibration
        # doppler_indata = windows.hann(len(doppler_indata)) * doppler_indata
        doppler_indata = np.squeeze(doppler_indata)
        doppler_indata = np.multiply(DOPPLER_HANN_WINDOW, doppler_indata)
        dfft = abs(np.fft.rfft(doppler_indata))

        # normalize doppler fft
        dfft /= np.amax(dfft)

        doppler_calibration_ffts.append(copy.deepcopy(dfft))

        if step > calibration_steps:
            subtracted_filtered = subtracted_fmcw[window_range]
            argmax = np.argmax(np.abs(subtracted_filtered))
            MEAN_WINDOW = 1
            mean_argmax = get_largest_n_mean(subtracted_filtered, MEAN_WINDOW)
            adjustment = window_range[0]
            #if not is_outlier(argmaxes, mean_argmax):
            argmaxes.append(mean_argmax)
            med_filtered = medfilt(argmaxes, 7)
            freqs = np.multiply(np.fft.rfftfreq(block_size), fs)
            argmax_distances = np.apply_along_axis(get_distance_from_peak, 0, med_filtered, window_range_start, median_peak_location)
            argmax_distances = np.apply_along_axis(idx_to_distance, 0, argmax_distances, freqs)
            #update_plot(argmax_distances)
            # li.set_xdata(np.arange(len(argmax_distances)))
            # li.set_ydata(argmax_distances)
            # # li2.set_xdata(np.arange(10000))
            # # li2.set_ydata(np.random.randn(10000))
            # plt.draw()
            # plt.pause(0.001)

            # new_argmax = med_filtered[-1]
            # print("MEAN: ", new_argmax)
            # freqs = np.fft.rfftfreq(len(subtracted))
            # freq = freqs[int(new_argmax)]
            # freq_in_hertz = abs(freq * fs)
            # distance = freq_in_hertz * 343 * .1 / 6000
            # distances.append(distance)

            # calculate doppler shift
            # global doppler_calibration_val
            # TODO inefficient since it only needs to be calculated once
            # not a huge deal though
            doppler_calibration_val = np.average(np.array(doppler_calibration_ffts), axis=0)

            subtracted_fft = np.subtract(dfft, doppler_calibration_val)

            # works better if we square the difference apparently
            subtracted_fft = np.multiply(subtracted_fft, subtracted_fft)

            DOPPLER_WINDOW = 50    # window around the tone frequency to scan for doppler shifts
            DOPPLER_FREQ   = 10000
            DOPPLER_TONE_IDX = find_nearest(np.fft.rfftfreq(doppler_indata.shape[0], d=1/SAMPLE_RATE), DOPPLER_FREQ)

            DOPPLER_WINDOW_BEGIN = int(DOPPLER_TONE_IDX - DOPPLER_WINDOW // 2)
            subtracted_fft = subtracted_fft[DOPPLER_WINDOW_BEGIN:
                                            DOPPLER_WINDOW_BEGIN + DOPPLER_WINDOW]

            # gaussian smoothing -- NEEDS TUNING
            subtracted_fft = ndimage.gaussian_filter1d(subtracted_fft, GAUSSIAN_SMOOTHING_SIGMA)

            global doppler_velocity
            # NOTE: negative because + value is getting closer to speaker
            new_velocity = -(np.average(np.arange(len(subtracted_fft)), weights=subtracted_fft) - DOPPLER_WINDOW / 2)

            # clamp to 0 if it's under a certain threshold
            if abs(new_velocity) < DOPPLER_VELOCITY_THRESHOLD:
                new_velocity = 0

            # do some exponential smooothing
            doppler_velocity = doppler_velocity * (1 - DOPPLER_SMOOTHING_FACTOR) + new_velocity * DOPPLER_SMOOTHING_FACTOR

            global doppler_velocities, doppler_distances
            doppler_velocities = np.hstack((doppler_velocities, np.array([doppler_velocity])))
            doppler_distances = np.hstack((doppler_distances, np.array([(doppler_distances[-1] if len(doppler_distances) else 0) + doppler_velocity])))
            interpolated_distance = get_position_update(argmax_distances[max(len(argmax_distances)-5, 0):], doppler_velocities[max(len(doppler_velocities) - 5, 0):])
            kalman_distances.append(interpolated_distance.item())
            #print(kalman_distances)
            # print(np.array(kalman_distances).shape)
            # print(np.array(kalman_distances)[:, 0, 0])

        # if previous is None:
        #     previous = np.copy(fft)
        # subtracted = np.subtract(fft, previous)
        # previous = np.copy(fft)
        # peak = np.argmax(subtracted)
        
        # if keep_going:
        #     return True
        # else:
        #     return False
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
CONST_FREQ = 10000
CHUNK = 4096
tone_time = 20#4096 / 48000
t = np.linspace(0, tone_time, int(tone_time*fs))
TONE = chirp(t, f0=CONST_FREQ, f1=CONST_FREQ, t1=10, method='linear').astype(np.float32)







T = .1
t = np.linspace(0, T, int(T*fs), endpoint=False)
w = chirp(t, f0=17000, f1=23000, t1=T, method='linear').astype(np.float32)
scaled = None
for i in range(200):
    if scaled is None:
        scaled = np.array(w)
    else:
        scaled = np.concatenate((scaled,w))

scaled = np.add(scaled / 2, TONE / 2)
# out_tone = np.zeros(max(len(TONE), len(scaled)))
# for x in range(len(out_tone)):
#     if x < len(TONE) and x < len(scaled):
#         out_tone[x] = TONE[x] + scaled[x]
#     elif x < len(TONE):
#         out_tone[x] = TONE[x]
#     else:
#         out_tone[x] = scaled[x]

# scaled = np.copy(out_tone)

#scaled = butter_bandpass_filter(scaled, 9900, 10100, 48000) #this works to leave only the constant tone
#scaled = butter_bandstop_filter(scaled, 9900, 10100, 48000) #this works to only get the chirps
#scaled = butter_bandstop_filter(scaled, 17000, 23000, 48000)

#scaled = np.add(scaled, TONE)

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
        update_plot()
    while not event.is_set():
        update_plot()
    event.wait()
    print("done")
    event.wait()
    # print(distances)
    # plt.plot(argmaxes[3:])
    # plt.show()
    
    # plt.plot(moving_average(argmax_distances[3:], 7))
    plt.show()
