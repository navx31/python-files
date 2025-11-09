import numpy as np
# Inserting into a 1D array
arr=np.array([20,30,40,50,60,70])
new_arr=np.insert(arr,2,35,axis=0)
print("Original array:", arr)
print("Array after insertion:", new_arr)

# Inserting into a 2D array
arr_2d=np.array([[1,2],[4,5]])
new_arr_2d=np.insert(arr_2d,1,[6,7],axis=0)
new_arr_2d_col=np.insert(arr_2d,2,[6,7],axis=1)
print("Original 2D array:\n", arr_2d)
print("2D array after row insertion:\n", new_arr_2d)
print("2D array after column insertion:\n", new_arr_2d_col)

#appending to a 1D array
arr_app=np.array([1,2,3])
new_arr_app=np.append(arr_app,[4,5,6],axis=0)
print("Original array for appending:", arr_app)
print("Array after appending:", new_arr_app)

#appending to a 2D array
arr_app_2d=np.array([[1,2,3],[4,5,6]])
new_arr_app_2d=np.append(arr_app_2d,[[7,8,9]],axis=0)
new_arr_app_2d_col=np.append(arr_app_2d,[[7],[8]],axis=1)
print("Original 2D array for appending:\n", arr_app_2d)
print("2D array after row appending:\n", new_arr_app_2d)
print("2D array after column appending:\n", new_arr_app_2d_col)

#concatenating arrays
arr1_concat=np.array([1,2,3])
arr2_concat=np.array([4,5,6])
concat_arr=np.concatenate((arr1_concat,arr2_concat),axis=0)
print("Concatenated array:", concat_arr)

#deleting from a 1D array
arr_del=np.array([10,20,30,40,50])
new_arr_del=np.delete(arr_del,2,axis=0)
print("Original array for deletion:", arr_del)
print("Array after deletion:", new_arr_del)

#deleting from a 2D array
arr_del_2d=np.array([[1,2,3],[4,5,6],[7,8,9]])
new_arr_del_2d=np.delete(arr_del_2d,1,axis=0)
new_arr_del_2d_col=np.delete(arr_del_2d,2,axis=1)
print("Original 2D array for deletion:\n", arr_del_2d)
print("2D array after row deletion:\n", new_arr_del_2d)
print("2D array after column deletion:\n", new_arr_del_2d_col)

#stacking arrays
arr_stack1=np.array([1,2,3])
arr_stack2=np.array([4,5,6])
vstack_arr=np.vstack((arr_stack1,arr_stack2))
hstack_arr=np.hstack((arr_stack1,arr_stack2))
print("Vertically stacked array:\n", vstack_arr)
print("Horizontally stacked array:", hstack_arr)

#stacking 2D arrays
arr_stack1_2d=np.array([[1,2,3],[4,5,6]])
arr_stack2_2d=np.array([[7,8,9],[10,11,12]])
vstack_arr_2d=np.vstack((arr_stack1_2d,arr_stack2_2d))
hstack_arr_2d=np.hstack((arr_stack1_2d,arr_stack2_2d))
print("Vertically stacked 2D array:\n", vstack_arr_2d)
print("Horizontally stacked 2D array:\n", hstack_arr_2d)    

#splitting arrays
arr_split=np.array([1,2,3,4,5,6])
split_arr=np.array_split(arr_split,3)
print("Original array for splitting:", arr_split)
print("Array after splitting into 3 parts:", split_arr)

#splitting 2D arrays
arr_split_2d=np.array([[1,2,3,4],[5,6,7,8]])
split_arr_2d=np.array_split(arr_split_2d,2,axis=0)
split_arr_2d_col=np.array_split(arr_split_2d,2,axis=1)
print("Original 2D array for splitting:\n", arr_split_2d)
print("2D array after splitting into 2 parts along rows:", split_arr_2d)
print("2D array after splitting into 2 parts along columns:", split_arr_2d_col)
