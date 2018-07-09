'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# setting channels to be last on tf backend
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# changing dtype from int into float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# normalize data so all values are in range [0-1]
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# view image in dataset
plt.imshow(x_test[0].reshape(img_rows, img_cols))
#plt.show()

# convert class vectors to binary class matrices (one hot encoded)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# define a sequential model
model = Sequential()

# CONV 32, f=3, s=1, p=0 : 28x28x1 -> 26x26x32
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

# CONV 64, f=3, s=1, p=0 : 26x26x32 -> 24x24x64
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

# MAXPOOL f=2 s=2 : 24x24x64 -> 12x12x64
model.add(MaxPooling2D(pool_size=(2, 2)))

# DROPOUT 12x12x64 -> 12x12x64
model.add(Dropout(0.25))

# FLATTEN : 12x12x64 -> 9216
model.add(Flatten())

# DENSE 9216 -> 128
model.add(Dense(128, activation='relu'))

# DROPOUT 128 -> 128
model.add(Dropout(0.5))

# SOFTMAX 10
model.add(Dense(num_classes, activation='softmax'))

# COMPILE
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# FIT
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# EVAL
model.summary()
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
