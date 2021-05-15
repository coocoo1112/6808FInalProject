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




def callback(indata, outdata, frames, time, status):
    try:
        data = q.get_nowait()
        
    except queue.Empty as e:
        print('Buffer is empty: increase buffersize?', file=sys.stderr)
        raise sd.CallbackAbort from e
    if any(indata):
        if len(data) == 4800:
            indatas.append(indata.reshape((block_size,)))
            outdatas.append(data.reshape((block_size,)))
        
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
    indatas = np.array(indatas)
    outdatas = np.array(outdatas)
    print(indatas.shape)
    print(outdatas.shape)
    #print(outdatas)
    pairs = np.hstack((np.array(indatas), np.array(outdatas)))
    np.save("experiment-1.npy", pairs)
    