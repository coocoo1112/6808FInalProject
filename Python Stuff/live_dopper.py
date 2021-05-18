import pyaudio
import numpy as np
import pylab
import matplotlib.pyplot as plt
from scipy.io import wavfile
import time
import sys
import scipy.signal as signal
import scipy.ndimage as ndimage
import copy


i=0
f,ax = plt.subplots(4)

# Prepare the Plotting Environment with random starting values
x = np.arange(10000)
y = np.random.randn(10000)

# Plot 0 is for raw audio data
li, = ax[0].plot(x, y)
ax[0].set_xlim(0,1000)
ax[0].set_ylim(0,500)
ax[0].set_title("testing")
# Plot 1 is for the FFT of the audio
li2, = ax[1].plot(x, y)
ax[1].set_xlim(0,30000)
ax[1].set_ylim(-100,100)
# ax[1].set_ylim(-1e-4,1e-4)
ax[1].set_title("Fast Fourier Transform")

li3, = ax[2].plot(x, y)
ax[2].set_xlim(0,400)
ax[2].set_ylim(-20, 20)
ax[2].set_title("Doppler velocity")

li4, = ax[3].plot(x, y)
ax[3].set_xlim(0,400)
ax[3].set_ylim(-200, 200)
ax[3].set_title("Doppler position (integrated)")
# Show the plot, but without blocking updates
plt.pause(0.01)
plt.tight_layout()

FORMAT = pyaudio.paFloat32 # We use 16bit format per sample
CHANNELS = 1
SAMPLE_RATE = 48000
CHUNK = 4096
# RECORD_SECONDS = 0.1
# WAVE_OUTPUT_FILENAME = "file.wav"

audio = pyaudio.PyAudio()

FREQ = 20000
T = 4096 / 48000
t = np.linspace(0, T, CHUNK)
TONE = signal.chirp(t, f0=FREQ, f1=FREQ, t1=4096 / 48000, method='linear').astype(np.int16)

# start Recording
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True)#,
                    #frames_per_buffer=CHUNK)

global keep_going
keep_going = True

# from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandstop(lowcut, highcut, fs, order=7):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='bandstop')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def butter_bandstop_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandstop(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

doppler_vels = np.array([])
doppler_distances = np.array([])
previous_fft = None
previous_ffts = []
all_ffts = None
steps = 0
calibration_ffts = []
calibration_val = None
velocity = 0

# find index in `array` with value closest to `value`
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def plot_data(in_data):
    # get and convert the data to float
    audio_data = np.fromstring(in_data, np.float32)
    # Fast Fourier Transform, 10*log10(abs) is to scale it to dB
    # and make sure it's not imaginary
    # dfft = 10.*np.log10(abs(np.fft.rfft(audio_data)))
    # filter the d
    audio_data = butter_bandpass_filter(audio_data, 19000, 21000, 48000, order=6)
    # audio_data = butter_bandstop_filter(audio_data, 19900, 20100, 48000, order=6)
    audio_data = signal.windows.hann(len(audio_data)) * audio_data
    dfft = 10.*np.log10(abs(np.fft.rfft(audio_data)))

    # normalize dfft
    # dfft /= np.linalg.norm(dfft)
    dfft /= np.amax(dfft)

    # and subtract it from the previous frame
    global previous_fft, all_ffts

    # HACK wait after first frame so we can do subtraction
    previous_fft = copy.deepcopy(dfft)

    # if previous_fft is None:
    #     return

    global steps, calibration_val
    steps += 1

    if steps < 5:
        calibration_ffts.append(previous_fft)
        return
    elif steps == 5:
        calibration_val = np.average(np.array(calibration_ffts), axis=0)

    # subtracted_fft = np.subtract(dfft, previous_fft)
    subtracted_fft = np.subtract(dfft, calibration_val)
    # subtracted_fft = dfft

    # print(len(dfft))
    # dfft = dfft[1650:1750]
    # peaks = signal.find_peaks(dfft)[0]

    # works better if we square the difference apparently
    subtracted_fft = np.multiply(subtracted_fft, subtracted_fft)
    
    DOPPLER_WINDOW = 50    # window around the tone frequency to scan for doppler shifts
    DOPPLER_FREQ   = 20000
    DOPPLER_TONE_IDX = find_nearest(np.fft.rfftfreq(audio_data.shape[0], d=1/SAMPLE_RATE), DOPPLER_FREQ)

    DOPPLER_WINDOW_BEGIN = int(DOPPLER_TONE_IDX - DOPPLER_WINDOW // 2)
    subtracted_fft = subtracted_fft[DOPPLER_WINDOW_BEGIN:
                                    DOPPLER_WINDOW_BEGIN + DOPPLER_WINDOW]

    # thresholding
    # print(np.amax(subtracted_fft))
    array_max = np.amax(subtracted_fft)
    # subtracted_fft = subtracted_fft > array_max * 0.2

    # # gaussian smoothing -- NEEDS TUNING
    GAUSSIAN_SMOOTHING_SIGMA = 4
    subtracted_fft = ndimage.gaussian_filter1d(subtracted_fft, GAUSSIAN_SMOOTHING_SIGMA)

    if all_ffts is None:
        all_ffts = np.array([subtracted_fft])
    else:
        all_ffts = np.vstack((all_ffts, np.array([subtracted_fft])))

    # print(peaks)
    # positive_peaks = negative_peaks = 0
    # velocity = (np.argmax(subtracted_fft) - 50) * 0.02
    print(np.average(np.arange(len(subtracted_fft)), weights=subtracted_fft))
    print(np.amax(subtracted_fft))
    print("ratio: ", np.amax(subtracted_fft) / np.amin(subtracted_fft))
    # if np.amax(subtracted_fft) > 0.1:

    global velocity
    new_velocity = np.average(np.arange(len(subtracted_fft)), weights=subtracted_fft) - DOPPLER_WINDOW / 2

    # do some exponential smooothing
    velocity = velocity * 0.5 + new_velocity * 0.5
    # clamp to 0 if it's under a certain threshold
    VELOCITY_THRESHOLD = 2.5
    if abs(velocity) < VELOCITY_THRESHOLD:
        velocity = 0

    # else:
        # velocity = 0

    global doppler_vels, doppler_distances
    # velocity = positive_peaks - negative_peaks
    doppler_vels = np.hstack((doppler_vels, np.array([velocity])))
    doppler_distances = np.hstack((doppler_distances, np.array([(doppler_distances[-1] if len(doppler_distances) else 0) + velocity])))
    # print(doppler_distances)

    # Force the new data into the plot, but without redrawing axes.
    # If uses plt.draw(), axes are re-drawn every time
    #print audio_data[0:10]
    #print dfft[0:10]
    #print
    li.set_xdata(np.arange(len(dfft)))
    li.set_ydata(dfft)
    li2.set_xdata(np.arange(len(subtracted_fft))*10.)
    # li2.set_ydata(10*np.log10(subtracted_fft))
    li2.set_ydata(subtracted_fft)
    li3.set_xdata(np.arange(len(doppler_vels)))
    li3.set_ydata(doppler_vels)
    li4.set_xdata(np.arange(len(doppler_distances)))
    li4.set_ydata(doppler_distances)

    # Show the updated plot, but without blocking
    plt.pause(0.001)
    if keep_going:
        return True
    else:
        return False

# Open the connection and start streaming the data
stream.start_stream()
#print "\n+---------------------------------+"
#print "| Press Ctrl+C to Break Recording |"
#print "+---------------------------------+\n"

# Loop so program doesn't end while the stream callback's
# itself for new data
while keep_going:
    try:
        plot_data(stream.read(CHUNK, exception_on_overflow = False))
    except KeyboardInterrupt:
        # keep_going=False
        print(all_ffts.shape)
        plt.figure(2)
        plt.pcolormesh(np.arange(all_ffts.shape[0]), np.arange(all_ffts.shape[1]), all_ffts.T, shading='gouraud')
        plt.pause(100)
        exit(0)
    except:
        pass

# Close up shop (currently not used because KeyboardInterrupt
# is the only way to close)
stream.stop_stream()
stream.close()

audio.terminate()
