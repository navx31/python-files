import numpy as np

arr1=np.zeros((3,3))
arr2=np.ones((2,3))
arr3=np.full((2,2),4)
arr4=np.arange(1,10,2)
arr5=np.eye(4)

print(arr1)
print(arr2)
print(arr3)
print(arr4)
print(arr5)

int_arr=np.array([[10,20,30,40,50],[15,25,35,45,55]])
print(int_arr.shape)
print(int_arr.size)
print(int_arr.ndim)
print(int_arr.dtype)
int_arr2=int_arr.astype(float)
print(int_arr2.dtype)

arry=np.array([21,32,43,54,65,76,87,98])
print(arry.mean())
print(arry.sum())
print(arry.max())
print(arry.min())
print(arry.std())
print(arry.var())

