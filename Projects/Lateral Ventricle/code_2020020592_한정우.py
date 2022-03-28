import os
import glob
import numpy as np
import nibabel as nib
from tensorflow import keras
from tensorflow.keras import layers
from keras.optimizers import Adam
from keras.layers import Dense, concatenate, Reshape
import tensorflow as tf
from PIL import Image
from model import *
from sklearn.model_selection import KFold


Train_Data = "/home/han/2020020592/Brain_CT_raw_file"
Label_Data = "/home/han/2020020592/ground_truth"


img_size = (256,256)
num_classes = 2
Epoch = 10

Train_Files = sorted(glob.glob(os.path.join(Train_Data, "*")))
Label_Files = sorted(glob.glob(os.path.join(Label_Data, "*")))

def data_loader(FileList):
    points_data = []

    for Files in FileList:
        raw_data = np.array(nib.load(Files).get_fdata())
        points_data.append(raw_data)

    return points_data


kf=KFold(n_splits=6,random_state=1,shuffle=False)


Train = np.array(data_loader(Train_Files))
Label = np.array(data_loader(Label_Files))

model = unet()


for i in range(Epoch):
    print("Epoch:", i+1 )
    for train_index, valid_index in kf.split(Train):

        Train_X, Train_Valid = Train[train_index], Train[valid_index]
        Label_X, Label_Valid = Label[train_index], Label[valid_index]

        model.fit(Train_X, Label_X, batch_size=25, epochs=1, verbose=1, validation_data=(Train_Valid, Label_Valid))

        del Train_X
        del Label_X

    score = model.evaluate(Train_Valid, Label_Valid, verbose=0)
    print('Train loss:', score[0])
    print('Train accuracy:', score[1])
    # model save
    model.save('Lateralventicle_v' + str(i) + '.h5')

    # score report
    f = open('report/ppp' + str(i) + "_score.txt", 'w')
    f.write('Train loss : ' + str(score[0]) + 'Train accuracy : ' +  str(score[1]))
    f.close()

    del Train_Valid
    del Label_Valid

del Train
del Label

Test_Data = "/home/han/2020020592/perfomance_test_file"
Test_Files = sorted(glob.glob(os.path.join(Test_Data, "*")))
Test = np.array(data_loader(Test_Files))

Result =model.predict(Test)

print(Result.shape)

Result = Result.reshape(100,256,256)


xx = np.max(Result)
yy = 255 / xx
Result = Result * yy

i = 0
for show in Result:
    i = i+1
    img = Image.fromarray(np.uint8(show))
    img = img.convert("L")

    if (i%20 == 0):
        img.show()
    img.save('img/'+str(i)+"_result.jpg")


def unet(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    #
    # conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    # conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    # drop5 = Dropout(0.5)(conv5)

    # up6 = Conv2D(512, 2, activation='relu', padding='same')(
    #     UpSampling2D(size=(2, 2))(drop4))
    # merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(drop4)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
