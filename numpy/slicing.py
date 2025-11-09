from ast import arg
import numpy as np

arg[1:4]     # elements from index 1 to 3

# 2D slicing
arr2d[0, :]  # first row: all columns
arr2d[:, 1]  # second column: all rows
arr2d[1:3, 1:3]  # subarray: rows 1-2, cols 1-2
