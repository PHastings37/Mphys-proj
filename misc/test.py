import numpy as np
array = np.random.randint(0, 100, size=(3, 3, 3))
np.pad(array, ((4,0), (0,0), (0,0)), "constant")
print(array)