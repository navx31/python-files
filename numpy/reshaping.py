import numpy as np
arr = np.arange(1, 10)

# reshape 1D → 2D
arr2d = arr.reshape(3, 3)   # 3x3 matrix

# 1D array of 12 elements
arr12 = np.arange(12)
arr12.reshape(4,3)  # 4 rows, 3 cols
arr12.reshape(3,4)  # 3 rows, 4 cols
