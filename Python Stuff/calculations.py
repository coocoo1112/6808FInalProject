import numpy as np


np_name = "experiment-1.npy"

arr = np.load(np_name)
print(arr.shape)
indata = arr[0]
outdata = arr[1]