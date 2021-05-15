import numpy as np


np_name = "experiment-1.npy"

arr = np.load(np_name)
print(arr.shape)