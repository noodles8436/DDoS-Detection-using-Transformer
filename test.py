import numpy as np

npz = np.load("./dataset_preprocessed/59.npz", allow_pickle=True)
x = npz['x']
y = npz['y']

print(x)
print(y)