# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 20:31:18 2019

@author: bill
"""

from keras.datasets import cifar10
from keras import utils as np_utils
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
X_train=x_train.reshape(x_train.shape[0],32,32,3).astype('float32')
x_Test = x_test.reshape(x_test.shape[0],32,32,3).astype('float32')

x_Train_norm = X_train / 255
x_Test_norm = x_Test / 255

y_TrainOneHot = np_utils.to_categorical(y_train)
y_TestOneHot = np_utils.to_categorical(y_test)

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Flatten,Conv2D,MaxPooling2D
model = Sequential()

model.add(Conv2D(32, kernel_size = (3,3), padding = 'same', 
                 input_shape = (32,32,3),
                 strides = (1,1), activation = 'relu'))
model.add(Conv2D(32, kernel_size=(3,3), padding='same', strides=(1,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Flatten())

model.add(Dense(400, input_dim=784))
model.add(Activation('relu'))
model.add(Dense(400, input_dim=784))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(width_shift_range=0.4, height_shift_range=0.6, zoom_range=0.5)
history = model.fit_generator(datagen.flow(x_Train_norm, y_TrainOneHot, batch_size=168),
                             samples_per_epoch=(len(x_train)*2), epochs=23, validation_data=(x_Test_norm, y_TestOneHot))

model.save('CIFAR-10.h5')

'''Access the loss and accuracy in every epoch'''
loss = history.history.get('loss')
acc  = history.history.get('categorical_accuracy')

''' Visualize the loss and accuracy of both models'''
import matplotlib.pyplot as plt
plt.figure(0)
plt.subplot(1,2,1)
plt.plot(range(len(loss)), loss,label='loss')
plt.title('Loss')

plt.subplot(1,2,2)
plt.plot(range(len(acc)), acc, label='accuracy')
plt.title('Accuracy')
plt.savefig('Keras Cifar10 model.png', dpi=300, format='png')
plt.show()
plt.close()
print('Result saved into Keras Cifar-10 model.png')

'''file download'''
from google.colab import files
files.download('Keras Cifar10 model.png')

