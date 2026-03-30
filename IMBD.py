from altair import sequence
from keras.datasets import imdb
(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)

word_index=imdb.get_word_index()                                 #Returns a dictionary mapping words to an integer index.
reverse_word_index=dict(
    (value,key)for key,value in word_index.items()               #Reverse the word index, mapping integer indices to words.
)
decode_review=''.join(
    reverse_word_index.get(i-3,'?')for i in train_data[0]        #Decode the review back to English words. The indices are offset by 3 because 0, 1, and 2 are reserved indices for "padding", "start of sequence", and "unknown".
)

#preparing the data
import numpy as np 
def vectorize_sequences(sequences,dimension=10000):              #Vectorize the sequences. This function will return a 2D numpy array of shape (len(sequences), dimension). The i-th row will be the vectorized representation of the i-th sequence, which will have 1s in the positions corresponding to the indices in the sequence and 0s elsewhere.
    results=np.zeros((len(sequences),dimension))                #Create an all-zero matrix of shape (len(sequences), dimension)
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1                                   #Set specific indices of results[i] to 1s
    return results
x_train=vectorize_sequences(train_data)                         #Vectorize the training data
x_test=vectorize_sequences(test_data)                            #Vectorize the test data

y_train=np.asarray(train_labels).astype('float32')           #Vectorize the labels. We will use one-hot encoding for the labels, which means that we will create a 2D numpy array of shape (len(labels), num_classes) where each row is a one-hot vector representing the class label.
y_test=np.asarray(test_labels).astype('float32')

from keras import models
from keras import layers 
model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))   #Add a densely-connected layer with 16 units to the model, followed by a ReLU activation function. The input shape is (10000,) because we are using one-hot encoding for the input data.
model.add(layers.Dense(16,activation='relu'))                        #Add another densely-connected layer with 16 units and a ReLU activation function.
model.add(layers.Dense(1,activation='sigmoid'))                       #Add a densely-connected layer with 1 unit and a sigmoid activation function
model.compile(
         optimizer='rmsprop',
         loss='binary_crossentropy',
         metrics=['accuracy'])   #Compile the model with the RMSprop optimizer, binary crossentropy loss function, and accuracy metric.

x_val=x_train[:10000]                         #Set aside a validation set. We will use the first 10,000 samples of the training data as a validation set and the remaining samples as the actual training data.
partial_x_train=x_train[10000:]
y_val=y_train[:10000]
partial_y_train=y_train[10000:]

history=model.fit(
    partial_x_train,
    partial_y_train,
    epochs=4,
    batch_size=512,
    validation_data=(x_val,y_val),
    verbose=1
)   #Train the model for 20 epochs with a batch size of 512, and validate the model on the validation set at the end of each epoch. 
result=model.evaluate(x_test,y_test)   #Evaluate the model on the test data and print the results.
print(result)
history_dict=history.history
history_dict.keys()   #Get the keys of the history dictionary, which contains the training and validation loss and accuracy for each epoch.
['loss', 'accuracy', 'val_loss', 'val_accuracy']
import matplotlib.pyplot as plt
history_dict=history.history
loss_values=history_dict['loss']   #Get the training loss values for each epoch.
val_loss_values=history_dict['val_loss']   #Get the validation loss values for each epoch.
epochs=range(1,len(loss_values)+1)   #Get the range of epochs for plotting.
plt.plot(epochs,loss_values,'bo',label='Training loss')   #Plot the training loss values as blue dots.
plt.plot(epochs,val_loss_values,'b',label='Validation loss')   #Plot the validation loss values as a blue line.
plt.title('Training and validation loss')   #Set the title of the plot.
plt.xlabel('Epochs')   #Set the x-axis label.
plt.ylabel('Loss')   #Set the y-axis label.
plt.legend()   #Add a legend to the plot.
plt.show()   #Display the plot.