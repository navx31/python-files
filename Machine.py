from keras.datasets import mnist 
import numpy as np 
import pandas as pd 
from keras import models
from keras import layers



from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape) 
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255 
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
model = models.Sequential()
model.add(layers.Input(shape=(784,)))   # or Flatten(input_shape=(784,))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10, batch_size=128, verbose=1)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)