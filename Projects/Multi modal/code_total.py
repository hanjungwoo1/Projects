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
#os.environ["CUDA_VISIBLE_DEVICES"] = "0";


input_size = int(100)

c1_train = np.loadtxt(fname = "data/total/c_train1.txt", dtype=str, delimiter=' ')
x1_train = np.loadtxt(fname = "data/total/x_train1.txt", dtype=str, delimiter=' ')
y1_train = np.loadtxt(fname = "data/total/y_train1.txt", dtype=str, delimiter=' ')
z1_train = np.loadtxt(fname = "data/total/z_train1.txt", dtype=str, delimiter=' ')
s1_train = np.loadtxt(fname = "data/total/s_train1.txt", dtype=str, delimiter=' ')

c1_test = np.loadtxt(fname = "data/total/c_test1.txt", dtype=str, delimiter=' ')
x1_test = np.loadtxt(fname = "data/total/x_test1.txt", dtype=str, delimiter=' ')
y1_test = np.loadtxt(fname = "data/total/y_test1.txt", dtype=str, delimiter=' ')
z1_test = np.loadtxt(fname = "data/total/z_test1.txt", dtype=str, delimiter=' ')
s1_test = np.loadtxt(fname = "data/total/s_test1.txt", dtype=str, delimiter=' ')

c2_train = np.loadtxt(fname = "data/total/c_train2.txt", dtype=str, delimiter=' ')
x2_train = np.loadtxt(fname = "data/total/x_train2.txt", dtype=str, delimiter=' ')
y2_train = np.loadtxt(fname = "data/total/y_train2.txt", dtype=str, delimiter=' ')
z2_train = np.loadtxt(fname = "data/total/z_train2.txt", dtype=str, delimiter=' ')
s2_train = np.loadtxt(fname = "data/total/s_train2.txt", dtype=str, delimiter=' ')

c2_test = np.loadtxt(fname = "data/total/c_test2.txt", dtype=str, delimiter=' ')
x2_test = np.loadtxt(fname = "data/total/x_test2.txt", dtype=str, delimiter=' ')
y2_test = np.loadtxt(fname = "data/total/y_test2.txt", dtype=str, delimiter=' ')
z2_test = np.loadtxt(fname = "data/total/z_test2.txt", dtype=str, delimiter=' ')
s2_test = np.loadtxt(fname = "data/total/s_test2.txt", dtype=str, delimiter=' ')

c3_train = np.loadtxt(fname = "data/total/c_train3.txt", dtype=str, delimiter=' ')
x3_train = np.loadtxt(fname = "data/total/x_train3.txt", dtype=str, delimiter=' ')
y3_train = np.loadtxt(fname = "data/total/y_train3.txt", dtype=str, delimiter=' ')
z3_train = np.loadtxt(fname = "data/total/z_train3.txt", dtype=str, delimiter=' ')
s3_train = np.loadtxt(fname = "data/total/s_train3.txt", dtype=str, delimiter=' ')

c3_test = np.loadtxt(fname = "data/total/c_test3.txt", dtype=str, delimiter=' ')
x3_test = np.loadtxt(fname = "data/total/x_test3.txt", dtype=str, delimiter=' ')
y3_test = np.loadtxt(fname = "data/total/y_test3.txt", dtype=str, delimiter=' ')
z3_test = np.loadtxt(fname = "data/total/z_test3.txt", dtype=str, delimiter=' ')
s3_test = np.loadtxt(fname = "data/total/s_test3.txt", dtype=str, delimiter=' ')

#### 
c_train = concatenate([c1_train, c2_train], axis= 0)
c_train = concatenate([c_train, c3_train], axis= 0)

del c1_train
del c2_train
del c3_train

c_test = concatenate([c1_test, c2_test], axis= 0)
c_test = concatenate([c_test, c3_test], axis= 0)

del c1_test
del c2_test
del c3_test

####x
x_train = concatenate([x1_train, x2_train], axis= 0)
x_train = concatenate([x_train, x3_train], axis= 0)

del x1_train
del x2_train
del x3_train

x_test = concatenate([x1_test, x2_test], axis= 0)
x_test = concatenate([x_test, x3_test], axis= 0)

del x1_test
del x2_test
del x3_test

####y
y_train = concatenate([y1_train, y2_train], axis= 0)
y_train = concatenate([y_train, y3_train], axis= 0)

del y1_train
del y2_train
del y3_train

y_test = concatenate([y1_test, y2_test], axis= 0)
y_test = concatenate([y_test, y3_test], axis= 0)

del y1_test
del y2_test
del y3_test

####z
z_train = concatenate([z1_train, z2_train], axis= 0)
z_train = concatenate([z_train, z3_train], axis= 0)

del z1_train
del z2_train
del z3_train

z_test = concatenate([z1_test, z2_test], axis= 0)
z_test = concatenate([z_test, z3_test], axis= 0)

del z1_test
del z2_test
del z3_test

####s
s_train = concatenate([s1_train, s2_train], axis= 0)
s_train = concatenate([s_train, s3_train], axis= 0)

del s1_train
del s2_train
del s3_train

s_test = concatenate([s1_test, s2_test], axis= 0)
s_test = concatenate([s_test, s3_test], axis= 0)

del s1_test
del s2_test
del s3_test


c_train = np.array(c_train, dtype = float)
x_train = np.array(x_train, dtype = float)
y_train = np.array(y_train, dtype = float)
z_train = np.array(z_train, dtype = float)
s_train = np.array(s_train, dtype = float)

c_test = np.array(c_test, dtype = float)
x_test = np.array(x_test, dtype = float)
y_test = np.array(y_test, dtype = float)
z_test = np.array(z_test, dtype = float)
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

Input_shape_c = Input(shape=(4))
Input_shape_x = Input(shape=(input_size, 1))
Input_shape_y = Input(shape=(input_size, 1))
Input_shape_z = Input(shape=(input_size, 1))


c_train = c_train.reshape(c_train.shape[0], 4)
c_test = c_test.reshape(c_test.shape[0], 4)

x_train = x_train.reshape(x_train.shape[0], input_size, 1)
x_test = x_test.reshape(x_test.shape[0], input_size, 1)

y_train = y_train.reshape(y_train.shape[0], input_size, 1)
y_test = y_test.reshape(y_test.shape[0], input_size, 1)

z_train = z_train.reshape(z_train.shape[0], input_size, 1)
z_test = z_test.reshape(z_test.shape[0], input_size, 1)

layer2 = Conv1D(32, 4, activation='relu')(Input_shape_x)
layer2 = BatchNormalization()(layer2)
layer2 = Conv1D(32, 4, activation='relu')(layer2)
layer2 = BatchNormalization()(layer2)
layer2_out = AveragePooling1D(pool_size=2)(layer2)

layer3 = Conv1D(32, 4, activation='relu')(Input_shape_y)
layer3 = BatchNormalization()(layer3)
layer3 = Conv1D(32, 4, activation='relu')(layer3)
layer3 = BatchNormalization()(layer3)
layer3_out = AveragePooling1D(pool_size=2)(layer3)

layer4 = Conv1D(32, 4, activation='relu')(Input_shape_z)
layer4 = BatchNormalization()(layer4)
layer4 = Conv1D(32, 4, activation='relu')(layer4)
layer4 = BatchNormalization()(layer4)
layer4_out = AveragePooling1D(pool_size=2)(layer4)


layer = concatenate([layer2_out, layer3_out, layer4_out])
layer = Reshape((47, 96, 1))(layer)


layer = conv1_layer(layer)
layer = conv2_layer(layer)
layer = conv3_layer(layer)
layer = conv4_layer(layer)
layer = conv5_layer(layer)
layer = GlobalAveragePooling2D()(layer)

layer = concatenate([layer, Input_shape_c])

layer = Dense(256)(layer)
layer = BatchNormalization()(layer)
layer = Activation('relu')(layer)

out = Dense(num_classes, activation='softmax')(layer)


model = Model(inputs=[Input_shape_c, Input_shape_x, Input_shape_y, Input_shape_z], outputs=out)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("==start==")

for i in range(1, 41):
    print(str(i)+'번째 start')

    model.fit([c_train, x_train, y_train, z_train], s_train, batch_size=batch_size, epochs=epochs, verbose=1)
    score = model.evaluate([c_test, x_test, y_test, z_test], s_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    #model.save('model/total_v'+str(i)+'.h5')

    f = open('report/total' + str(i) + "_score.txt", 'w')
    f.write(str(score[1]))
    f.close()
    print(str(i) + '번째 end')


print("==end==")

print("==classfication== start")

score = model.evaluate([c_test, x_test, y_test, z_test], s_test, verbose=0)
s_test_class = np.argmax(s_test, axis=1)
s_pred = np.argmax(model.predict([c_test, x_test, y_test, z_test]), axis=1)

result = classification_report(s_test_class, s_pred)
f = open('report/total_classificaiton_report.txt', 'w')
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
    f = open('report/total' + str(k) + "_range_accuracy.txt", 'w')
    f.write(result)
    f.close()
print("==Range== end")



#################################

x = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
     105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200
     ]

z=[]

for i in range(1,41):
    name = "report/total" + str(i) + "_score.txt"

    f = open(name, 'r')
    line = f.readline()
    line = line[0:7]
    z.append(float(line))
    f.close()


plt.plot(x, z, label='Multi_total')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('Accuracy in Epoch')
plt.legend()
plt.savefig('result/total_acc_last.png', dpi=300)
plt.show()


