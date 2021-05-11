import sounddevice as sd
import numpy as np
from scipy.signal import chirp
import queue
import threading
import sys
import shutil
import math
import datetime

#global scaled
buff_size = 40
block_size = 882
q = queue.Queue(maxsize=buff_size)
event = threading.Event()
gain = 10
low, high = 17000, 23000
block_duration = 20 #in milliseconds
start = datetime.datetime.now()
print(sd.query_devices())


def callback(indata, outdata, frames, time, status):
    try:
        data = q.get_nowait()
    except queue.Empty as e:
        print('Buffer is empty: increase buffersize?', file=sys.stderr)
        raise sd.CallbackAbort from e
    if any(indata):
        global previous
        #print(datetime.datetime.now() - start)
        if previous is None:
            subtracted_fft = np.fft.rfft(indata[:, 0], n=fftsize)
            previous = subtracted_fft
        else:
            fft = np.fft.rfft(indata[:, 0], n=fftsize)
            subtracted_fft = np.subtract(fft, previous)
            previous = fft
        
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
        print(len(data), len(outdata))
        outdata[:len(data)] = data.reshape(len(data), 1)
        outdata[len(data):] = np.zeros(((len(outdata) - len(data), 1)))
        raise sd.CallbackStop
    else:
        outdata[:] = data.reshape((block_size, 1))

def in_callback(indata, frames, time, status):
        if status:
            text = ' ' + str(status) + ' '
            print('\x1b[34;40m', text.center(columns, '#'),
                  '\x1b[0m', sep='')
        if any(indata):
            print(np.fft.rfft(indata[:, 0], n=fftsize))
            # magnitude = np.abs(np.fft.rfft(indata[:, 0], n=fftsize))
            # magnitude *= gain / fftsize
            # line = (gradient[int(np.clip(x, 0, 1) * (len(gradient) - 1))]
            #         for x in magnitude[low_bin:low_bin + columns])
            #print(*line, sep='', end='\x1b[0m\n')
        else:
            print('no input')



# try:
#     with sd.Stream(device=(0,1), samplerate=fs, dtype='float32', latency='low', channels=(1,2), callback=callback, blocksize=0):
#         input()
# except KeyboardInterrupt:
#     pass

try:
    columns, _ = shutil.get_terminal_size()
except AttributeError:
    columns = 80

fs = 44100
fs = int(sd.query_devices(1, 'input')['default_samplerate'])
print(fs)
T = .02
t = np.linspace(0, T, int(T*fs))
w = chirp(t, f0=17000, f1=23000, t1=T, method='linear').astype(np.float32)
scaled = np.int16(w/np.max(np.abs(w)) * 32767) 
scaled = w

for i in range(6):
    scaled = np.concatenate((scaled, scaled))

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
low_bin = math.floor(low / delta_f)   
previous = None

for _ in range(20):
    data = scaled[:min(block_size, len(scaled))]
    if len(data) == 0:
        break
    scaled = scaled[min(block_size, len(scaled)):]
    q.put_nowait(data)


with sd.Stream(device=(1,2), samplerate=fs, dtype='float32', latency='low', channels=(1,2), callback=callback, blocksize=block_size, finished_callback=event.set):
    timeout = block_size * buff_size / fs
    while len(data) != 0:
        data = scaled[:min(block_size, len(scaled))]
        scaled = scaled[min(block_size, len(scaled)):]
        q.put(data, timeout=timeout)
    event.wait()


# with sd.InputStream(device=0, channels=1, callback=in_callback,
#                         blocksize=block_size,#int(fs * block_duration / 1000),
#                         samplerate=fs):
#     while True:
#         response = input()
#         if response in ('', 'q', 'Q'):
#             break





#output stream working

# try:
#     stream = sd.OutputStream(device=1, samplerate=fs, dtype='float32', latency='low', channels=1, callback=out_callback, blocksize=block_size, finished_callback=event.set)
#     with stream:
#         print("starting")
#         timeout = block_size * buff_size / fs
#         while len(data) != 0:
#             data = scaled[:min(block_size, len(scaled))]
#             scaled = scaled[min(block_size, len(scaled)):]
#             q.put(data, timeout=timeout)
#         event.wait()

# except KeyboardInterrupt:
#         pass
