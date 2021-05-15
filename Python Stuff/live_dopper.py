import pyaudio
import numpy as np
import pylab
import matplotlib.pyplot as plt
from scipy.io import wavfile
import time
import sys
import seaborn as sn
import scipy.signal as signal


i=0
f,ax = plt.subplots(4)

# Prepare the Plotting Environment with random starting values
x = np.arange(10000)
y = np.random.randn(10000)

# Plot 0 is for raw audio data
li, = ax[0].plot(x, y)
ax[0].set_xlim(0,1000)
ax[0].set_ylim(-5000,5000)
ax[0].set_title("testing")
# Plot 1 is for the FFT of the audio
li2, = ax[1].plot(x, y)
ax[1].set_xlim(0,1000)
ax[1].set_ylim(-100,100)
ax[1].set_title("Fast Fourier Transform")

li3, = ax[2].plot(x, y)
ax[2].set_xlim(0,400)
ax[2].set_ylim(-20, 20)
ax[2].set_title("Doppler velocity")

li4, = ax[3].plot(x, y)
ax[3].set_xlim(0,400)
ax[3].set_ylim(-20, 20)
ax[3].set_title("Doppler position (integrated)")
# Show the plot, but without blocking updates
plt.pause(0.01)
plt.tight_layout()

FORMAT = pyaudio.paFloat32 # We use 16bit format per sample
CHANNELS = 1
RATE = 48000
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
                    rate=RATE,
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

def plot_data(in_data):
    # get and convert the data to float
    audio_data = np.fromstring(in_data, np.float32)
    # Fast Fourier Transform, 10*log10(abs) is to scale it to dB
    # and make sure it's not imaginary
    # dfft = 10.*np.log10(abs(np.fft.rfft(audio_data)))
    # filter the d
    print(audio_data.shape)
    audio_data = butter_bandstop_filter(audio_data, 19900, 20100, 48000, order=6)
    dfft = 10.*np.log10(abs(np.fft.rfft(audio_data)))

    # print(len(dfft))
    dfft = dfft[1650:1750]
    peaks = signal.find_peaks(dfft)[0]

    print(len(peaks[peaks < 44]))
    negative_peaks = len(peaks[peaks < 44])
    print(len(peaks[peaks > 70]))
    positive_peaks = len(peaks[peaks > 70])

    print(peaks)
    global doppler_vels, doppler_distances
    velocity = positive_peaks - negative_peaks
    print("HI")
    doppler_vels = np.hstack((doppler_vels, np.array([velocity])))
    doppler_distances = np.hstack((doppler_distances, np.array([(doppler_distances[-1] if len(doppler_distances) else 0) + velocity])))
    print(doppler_distances)

    # Force the new data into the plot, but without redrawing axes.
    # If uses plt.draw(), axes are re-drawn every time
    #print audio_data[0:10]
    #print dfft[0:10]
    #print
    li.set_xdata(np.arange(len(audio_data)))
    li.set_ydata(audio_data)
    li2.set_xdata(np.arange(len(dfft))*10.)
    li2.set_ydata(dfft)
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
        keep_going=False
    except:
        pass

# Close up shop (currently not used because KeyboardInterrupt
# is the only way to close)
stream.stop_stream()
stream.close()

audio.terminate()
