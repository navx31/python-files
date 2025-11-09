import numpy as np

arr = np.array([1, 2, 3, 4])
# Range arrays
arr1 = np.arange(0, 10, 2)      
arr2 = np.linspace(0, 1, 5)     
arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
print("1D array:", arr)
print("Range array using arange:", arr1)
print("Range array using linspace:", arr2)
print("2D array:\n", arr2d)

