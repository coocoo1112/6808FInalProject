import numpy as np


np_name = "experiment-static-1.npy"

arr = np.load(np_name)
print(arr.shape)
indata = arr[0]
outdata = arr[1]
zeros = np.zeros(len(indata[0]))

for i in range(len(indata) - 1):
    print(i)
    sub = np.subtract(indata[i], indata[i+1])
    assert np.allclose(zeros, sub)
