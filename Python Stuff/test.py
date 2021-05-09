import numpy as np
from scipy.signal import chirp
from scipy.io.wavfile import write, read
import pyaudio
import wave
import sys
import threading
import concurrent.futures




# m = np.max(np.abs(w))
# sigf32 = (w/m).astype(np.float32)
# write('sound.wav', fs, sigf32)
# print("done")

def sound(inp):
    array, fs = inp
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=len(array.shape), rate=fs, output=True)
    stream.write(array.tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()
    return "x"

def record(duration=3, fs=8000):
    nsamples = duration*fs
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True,
                    frames_per_buffer=nsamples)
    buffer = stream.read(nsamples)
    array = np.frombuffer(buffer, dtype='int16')
    stream.stop_stream()
    stream.close()
    p.terminate()
    return array


def play(data):
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt32  # 16 bits per sample
    channels = 1
    fs = 44100  # Record at 44100 samples per second
    seconds = 3
    filename = "output.wav"

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    output=True)
    
    for i in range(0, int(fs / chunk * seconds)):
        stream.write(data[i])
    
    #stream.write(data)
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()


def record_v2():
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt32  # 16 bits per sample
    channels = 1
    fs = 44100  # Record at 44100 samples per second
    seconds = 3
    filename = "output.wav"

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True,
                    output=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for 3 seconds
    #frames.append(stream.read(fs * seconds))

    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        stream.write(data)
        frames.append(data)

    # Stop and close the stream 
    #stream.write(frames[0])
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')
    return frames

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()


# Set chunk size of 1024 samples per data frame
chunk = 1024  

# Open the sound file 
# wf = wave.open(filename, 'rb')
print("step1")

# # Create an interface to PortAudio
# print("step2")

# # Open a .Stream object to write the WAV file to
# # 'output = True' indicates that the sound will be played rather than recorded
# stream = p.open(format=pyaudio.paFloat32,
#                          channels=len(w.shape),
#                          rate=fs,
#                          output=True
#                          )

# stream.write(w)

# print("step3")

# # Read data in chunks
# data = wf.readframes(chunk)
# print('step4')

# # Play the sound by writing the audio data to the stream
# i = 1
# while data != '':
#     print('chunk: {}'.format(i))
#     stream.write(data)
#     data = wf.readframes(chunk)
#     i += 1

# # Close and terminate the stream
# stream.close()
# p.terminate()
# print("done")

def thread_function():
    for i in range(100):
        print(i)


if __name__ == "__main__":

    # x = threading.Thread(target=thread_function, args=())
    # y = threading.Thread(target=thread_function, args=())
    # x.start()
    # y.start()


    #sys.exit()
    t = np.linspace(0, 10, 1500)
    fs = 44100
    T = 1
    t = np.linspace(0, T, T*fs)
    w = chirp(t, f0=1000, f1=3000, t1=T, method='linear').astype(np.float32)

    filename = 'myfile.wav'
    print(w[0])
    scaled = np.int16(w/np.max(np.abs(w)) * 32767) 
    #x = threading.Thread(target=sound, args=(scaled, fs))
   # y = threading.Thread
    #print(x.start())
    with concurrent.futures.ThreadPoolExecutor() as executor:
        x = threading.Thread(target=sound, args=([scaled, fs],))
        x.start()
        #x = executor.submit(sound, (scaled, fs))
        y = executor.submit(record_v2)
        return_value = y.result()
        print(return_value)
    #sound(scaled, fs)
    sys.exit()
    print(type(scaled[0]))

    # write(filename, fs, scaled)
    # wf = wave.open(filename, 'rb')


    # fs1, data = read(filename)
    # print(fs1)
    # #data = data * 10
    # test_data = np.array(data)
    # for i in range(3):
    #     test_data = np.concatenate((test_data, data))
    #sound(test_data, fs1)
    print("recording")
    test = record_v2()
    play(test)



