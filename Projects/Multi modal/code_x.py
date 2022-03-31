#!/usr/bin/python
#-*-coding:utf-8-*-

import numpy as np
import keras
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, concatenate, Reshape
from keras.layers import Conv2D, Input, Activation, Conv1D
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, MaxPooling1D
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import tensorflow as tf
from scipy.stats import skew, kurtosis
import scipy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Add, AveragePooling1D
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint
import os


# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "1";


input_size = int(100)

x_train = np.loadtxt(fname = "data/x_train1.txt", dtype=str, delimiter=' ')
s_train = np.loadtxt(fname = "data/s_train1.txt", dtype=str, delimiter=' ')

x_test = np.loadtxt(fname = "data/x_test1.txt", dtype=str, delimiter=' ')
s_test = np.loadtxt(fname = "data/s_test1.txt", dtype=str, delimiter=' ')


x_train = np.array(x_train, dtype = float)
s_train = np.array(s_train, dtype = float)

x_test = np.array(x_test, dtype = float)
s_test = np.array(s_test, dtype = float)



def conv1_layer(x):
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)

    return x

def conv2_layer(x):
    x = MaxPooling2D((3, 3), 2)(x)

    shortcut = x

    for i in range(3):
        if (i == 0):
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x


def conv3_layer(x):
    shortcut = x

    for i in range(4):
        if (i == 0):
            x = Conv2D(128, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x


def conv4_layer(x):
    shortcut = x

    for i in range(6):
        if (i == 0):
            x = Conv2D(256, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(1024, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x


def conv5_layer(x):
    shortcut = x

    for i in range(3):
        if (i == 0):
            x = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(2048, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x


####################################################################################################

batch_size = 100
num_classes = 100
epochs = 5


# label 데이터 분류
s_train = keras.utils.to_categorical(s_train, num_classes)
s_test = keras.utils.to_categorical(s_test, num_classes)

Input_shape_x = Input(shape=(input_size, 1))

x_train = x_train.reshape(x_train.shape[0], input_size, 1)
x_test = x_test.reshape(x_test.shape[0], input_size, 1)


layer = Conv1D(32, 4, activation='relu')(Input_shape_x)
layer = BatchNormalization()(layer)
layer = Conv1D(32, 4, activation='relu')(layer)
layer = BatchNormalization()(layer)
layer_out = AveragePooling1D(pool_size=2)(layer)

layer = Reshape((47, 32, 1))(layer_out)

layer = conv1_layer(layer)
layer = conv2_layer(layer)
layer = conv3_layer(layer)
layer = conv4_layer(layer)
layer = conv5_layer(layer)
layer = GlobalAveragePooling2D()(layer)

layer = Dense(256)(layer)
layer = BatchNormalization()(layer)
layer = Activation('relu')(layer)

out = Dense(num_classes, activation='softmax')(layer)



model = Model(inputs=Input_shape_x, outputs=out)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("==start==")

for i in range(1, 41):
    print(str(i)+'번째 start')

    model.fit(x_train, s_train, batch_size=batch_size, epochs=epochs, verbose=1)
    score = model.evaluate(x_test, s_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save('model/x_v'+str(i)+'.h5')

    f = open('report/x' + str(i) + "_score.txt", 'w')
    f.write(str(score[1]))
    f.close()
    print(str(i) + '번째 end')

print("==end==")



print("==classfication== start")

score = model.evaluate(x_test, s_test, verbose=0)
s_test_class = np.argmax(s_test, axis=1)
s_pred = np.argmax(model.predict(x_test), axis=1)

result = classification_report(s_test_class, s_pred)
f = open('report/x_classificaiton_report.txt', 'w')
f.write(result)
f.close()
print("==classfication== end")





print("==Range== start")
for k in range(1,6):
    i = 0
    sum = 0
    for value in s_pred:
        min_value = value - k
        max_value = value + k

        if ((min_value < s_test_class[i]) and (s_test_class[i] < max_value)):
            # print("성공")
            sum = sum + 1
        # else:
        #     print("실패")

        i = i + 1
    result = str(k)+ "range accuracy:" + str(sum / 78000)
    print("accuracy: " + result)
    f = open('report/x' + str(k) + "_range_accuracy.txt", 'w')
    f.write(result)
    f.close()
print("==Range== end")


