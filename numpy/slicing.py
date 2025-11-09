import numpy as np

matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

# Get the first row
row1 = matrix[0, :]
print(f"First row (matrix[0, :]): {row1}")

# Get the second column
col2 = matrix[:, 1]
print(f"Second column (matrix[:, 1]): {col2}")

# Get a sub-matrix (rows 0-1, columns 1-2)
sub_matrix = matrix[0:2, 1:3]
print(f"Sub-matrix (matrix[0:2, 1:3]):\n{sub_matrix}")

# Get every other row and every other column
stepped_slice = matrix[::2, ::2]
print(f"Stepped slice (matrix[::2, ::2]):\n{stepped_slice}")
