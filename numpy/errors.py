import numpy as np 

arr=np.array([10,20,np.nan,40,50,np.nan])
print("Original array:", arr)
print(np.isnan(arr))  # Check for NaN values

# Replace NaN with the mean of non-NaN values
arr=np.array([10,20,np.nan,40,50,np.nan])
cleared_arr=np.nan_to_num(arr, nan=100)
print(cleared_arr)

#infinite values
arr_inf=np.array([1,2,np.inf,4,-np.inf])
print(arr_inf)
cleared_arr_inf=np.nan_to_num(arr_inf, posinf=1e10, neginf=-1e10)
print(cleared_arr_inf)