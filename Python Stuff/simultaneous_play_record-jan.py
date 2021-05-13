from operator import sub
import sounddevice as sd
import numpy as np
from scipy.signal import chirp, spectrogram, find_peaks
import queue
import threading
import sys
import shutil
import math
import datetime
import scipy.io.wavfile as wav
import json
import matplotlib.pyplot as plt

#global scaled
buff_size = 40
block_size = 4800#960
q = queue.Queue(maxsize=buff_size)
event = threading.Event()
gain = 10
low, high = 17000, 23000#17000, 23000
block_duration = 20 #in milliseconds
start = datetime.datetime.now()
print(sd.query_devices())
total_result = None
distances = []
ffts = []

sound_data_received = []
sound_data_sent = []

def plot(fft):
    plt.plot(fft)
    plt.show()
    sys.exit()


def callback(indata, outdata, frames, time, status):
    try:
        data = q.get_nowait()
    except queue.Empty as e:
        print('Buffer is empty: increase buffersize?', file=sys.stderr)
        raise sd.CallbackAbort from e
    if any(indata):
        global previous
        global total_result
        global fs
        global distances
        global T
        #print(datetime.datetime.now() - start)
        multiplied = np.multiply(indata, outdata)
        multiplied = indata

        # Capturing the sound that is being sent 
        sound_data_sent.append(outdata)

        # Capturing the sound that is being received        
        sound_data_received.append(indata)

        # plot(indata)
        #print(indata.shape, outdata.shape, multiplied.shape)
        if previous is None:
            subtracted_fft = np.fft.rfft(multiplied[:, 0], n=fftsize)#indata
            previous = subtracted_fft
        else:
            fft = np.fft.rfft(multiplied[:, 0], n=fftsize)#indata
            subtracted_fft = np.subtract(fft, previous)
            previous = fft
            #plot(subtracted_fft)
            ffts.append(np.fft.rfft(data.reshape((block_size, 1))[:, 0], n=fftsize))
            
        if total_result is None:
            total_result = [subtracted_fft]
        else:
            total_result.append(subtracted_fft)
        
        freqs = np.fft.fftfreq(len(subtracted_fft))
        #print(subtracted_fft)
        idx = find_peaks(subtracted_fft)#np.argmax(np.abs(subtracted_fft))
        if len(idx[0]) != 0:
            idx = idx[0][0]
            #print(idx)
            freq = freqs[idx]
            freq_in_hertz = abs(freq * fs)
            print("test", freq_in_hertz)
            distance = freq_in_hertz * 343 * T / 6000#(high - low)
            #print(distance)
            distances.append(distance)
            magnitude = np.abs(subtracted_fft)#np.fft.rfft(indata[:, 0], n=fftsize))
            magnitude *= gain / fftsize
            line = (gradient[int(np.clip(x, 0, 1) * (len(gradient) - 1))]
                    for x in magnitude[low_bin:low_bin + columns])
            print(*line, sep='', end='\x1b[0m\n')
    else:
        print('no input')
    if len(data) < len(outdata):
        print(len(data), len(outdata))
        outdata[:len(data)] = data.reshape(len(data), 1)
        outdata[len(data):] = np.zeros(((len(outdata) - len(data), 1)))
        raise sd.CallbackStop
    else:
        outdata[:] = data.reshape((block_size, 1))
    


def out_callback(outdata, frames, time, status):
    print(frames)
    print(outdata.shape)
    try:
        data = q.get_nowait()
    except queue.Empty as e:
        print('Buffer is empty: increase buffersize?', file=sys.stderr)
        raise sd.CallbackAbort from e
    if len(data) < len(outdata):
        print("test")
        print(len(data), len(outdata))
        outdata[:len(data)] = data.reshape(len(data), 1)
        outdata[len(data):] = np.zeros(((len(outdata) - len(data), 1)))
        raise sd.CallbackStop
    else:
        
        temp = data.reshape((block_size, 1))
        outdata[:] = temp

def in_callback(indata, frames, time, status):
    global start
    # print(datetime.datetime.now()-start)
    # print(frames)
    # print(len(indata))
    if status:
        text = ' ' + str(status) + ' '
        print('\x1b[34;40m', text.center(columns, '#'),
                '\x1b[0m', sep='')
    if any(indata):
        global previous
        global total_result
        global fs
        global distances
        global T
        #print(datetime.datetime.now() - start)
        #multiplied = np.multiply(indata, outdata)
        multiplied = indata
        #print(indata.shape, outdata.shape, multiplied.shape)
        if previous is None:
            subtracted_fft = np.fft.rfft(multiplied[:, 0], n=fftsize)#indata
            previous = subtracted_fft
        else:
            fft = np.fft.rfft(multiplied[:, 0], n=fftsize)#indata
            subtracted_fft = np.subtract(fft, previous)
            previous = fft
            #plot(subtracted_fft)
            
        if total_result is None:
            total_result = [subtracted_fft]
        else:
            total_result.append(subtracted_fft)
        
        freqs = np.fft.fftfreq(len(subtracted_fft))
        idx = np.argmax(np.abs(subtracted_fft))#find_peaks(subtracted_fft)[0][0]#
        #print(idx)
        freq = freqs[idx]
        freq_in_hertz = abs(freq * fs)
        print(freq_in_hertz)
        distance = freq_in_hertz * 343 * T / (high - low)
        #print(distance)
        distances.append(distance)
        magnitude = np.abs(subtracted_fft)#np.fft.rfft(indata[:, 0], n=fftsize))
        magnitude *= gain / fftsize
        line = (gradient[int(np.clip(x, 0, 1) * (len(gradient) - 1))]
                for x in magnitude[low_bin:low_bin + columns])
        print(*line, sep='', end='\x1b[0m\n')
    else:
        print('no input')
    
    # fft = np.fft.rfft(indata[:, 0], n=fftsize)

    magnitude = np.abs(subtracted_fft)#np.fft.rfft(indata[:, 0], n=fftsize))
    magnitude *= gain / fftsize
    line = (gradient[int(np.clip(x, 0, 1) * (len(gradient) - 1))]
            for x in magnitude[low_bin:low_bin + columns])
    print(*line, sep='', end='\x1b[0m\n')

# def in_callback(indata, frames, time, status):
#         if status:
#             text = ' ' + str(status) + ' '
#             print('\x1b[34;40m', text.center(columns, '#'),
#                   '\x1b[0m', sep='')
#         if any(indata):
#             print(np.fft.rfft(indata[:, 0], n=fftsize))
#             # magnitude = np.abs(np.fft.rfft(indata[:, 0], n=fftsize))
#             # magnitude *= gain / fftsize
#             # line = (gradient[int(np.clip(x, 0, 1) * (len(gradient) - 1))]
#             #         for x in magnitude[low_bin:low_bin + columns])
#             #print(*line, sep='', end='\x1b[0m\n')
#         else:
#             print('no input')



# try:
#     with sd.Stream(device=(0,1), samplerate=fs, dtype='float32', latency='low', channels=(1,2), callback=callback, blocksize=0):
#         input()
# except KeyboardInterrupt:
#     pass

try:
    columns, _ = shutil.get_terminal_size()
except AttributeError:
    columns = 80

fs = 48000
fs = int(sd.query_devices(0, 'input')['default_samplerate'])
print(fs)
T = .1#.02
t = np.linspace(0, T, int(T*fs), endpoint=False)
w = chirp(t, f0=17000, f1=23000, t1=T, method='linear').astype(np.float32)
#scaled = np.int16(w/np.max(np.abs(w)) * 32767) 
scaled = None
# fs, scaled = wav.read("fmcw_chirp.wav")
# wav.write("idk1.wav", fs, scaled[:,0])
# print(fs)

for i in range(50):
    if scaled is None:
        scaled = np.array(w)
    else:
        scaled = np.concatenate((scaled,w))#(scaled, scaled))

colors = 30, 34, 35, 91, 93, 97
chars = ' :%#\t#%:'
gradient = []

for bg, fg in zip(colors, colors[1:]):
    for char in chars:
        if char == '\t':
            bg, fg = fg, bg
        else:
            gradient.append('\x1b[{};{}m{}'.format(fg, bg + 10, char))


delta_f = (high - low) / (columns - 1)
fftsize = math.ceil(fs / delta_f)
# fftsize = 1920
low_bin = math.floor(low / delta_f)   
previous = None

for _ in range(20):
    data = scaled[:min(block_size, len(scaled))]#,0]
    if len(data) == 0:
        break
    scaled = scaled[min(block_size, len(scaled)):]
    print(data.shape)
    print("loop")
    q.put_nowait(data)


with sd.Stream(device=(0,1), samplerate=fs, dtype='float32', latency='low', channels=(1,2), callback=callback, blocksize=block_size, finished_callback=event.set):
    timeout = block_size * buff_size / fs
    while len(data) != 0:
        data = scaled[:min(block_size, len(scaled))]#,0]
        scaled = scaled[min(block_size, len(scaled)):]
        q.put(data, timeout=timeout)
    
    # f, t, Sxx = spectrogram(np.array(total_result), fs)
    # plt.pcolormesh(t, f, Sxx, shading='gouraud')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
    # string = json.dumps(total_result)
    # with open("testidk.txt", 'w') as f:
    #     f.write(string)
    # test = json.loads(string)
    # print(type(test))
    # print(type(test[0]))
    event.wait()
    print(len(distances))
    print(min(distances))
    print(len(ffts))
    # for i in range (0, len(ffts), 5):
    #     plt.plot(ffts[i])
    #     plt.show()
    # plt.plot(distances)


    ##### Checking the shape and structure of the input data


    fft_of_recevied_data = np.fft.fft(sound_data_received[0])
    fft_of_sent_data = np.fft.fft(sound_data_sent[0])

    # Plotting the data that is being sent
    plt.plot(sound_data_received[0])
    plt.show()

    plt.plot(fft_of_recevied_data)
    plt.show()


    plt.plot(sound_data_sent[0])
    plt.show()

    plt.plot(fft_of_sent_data)
    plt.show()
# # print(fs)
# with sd.InputStream(device=0, channels=1, callback=in_callback,
#                         blocksize=int(fs * block_duration / 1000),
#                         samplerate=fs):
#     while True:
#         response = input()
#         if response in ('', 'q', 'Q'):
#             break





#output stream working

# try:
#     stream = sd.OutputStream(device=1, samplerate=fs, dtype='float32', latency='low', channels=2, callback=out_callback, blocksize=block_size, finished_callback=event.set)
#     with stream:
#         print("starting")
#         timeout = block_size * buff_size / fs
#         while len(data) != 0:
#             print("hmmm")
#             data = scaled[:min(block_size, len(scaled))]
#             scaled = scaled[min(block_size, len(scaled)):]
#             q.put(data, timeout=timeout)
#         event.wait()

# except KeyboardInterrupt:
#         pass

