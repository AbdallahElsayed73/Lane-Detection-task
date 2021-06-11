import numpy as np
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras import backend as keras
from keras import optimizers
import cv2 

inputs = Input((128,128,3))
conv128 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
pool128 = MaxPool2D(pool_size=(2,2))(conv128)

conv64 = Conv2D(32,3,activation='relu', padding='same')(pool128)
pool64 = MaxPool2D(pool_size=(2,2))(conv64)

conv32 = Conv2D(32,3,activation='relu', padding='same')(pool64)
pool32 = MaxPool2D(pool_size=(2,2))(conv32)


conv16 = Conv2D(32,3,activation='relu', padding='same')(pool32)
pool16 = MaxPool2D(pool_size=(2,2))(conv16)


conv8 = Conv2D(32,3,activation='relu', padding='same')(pool16)
conv8 = Conv2D(32,3,activation='relu', padding='same')(conv8)

up16 = UpSampling2D(size=(2,2))(conv8)
up16 = Conv2D(32,3,activation='relu', padding='same')(up16)

conc16 = concatenate([conv16,up16], axis = 3)
up32 = UpSampling2D(size=(2,2))(conc16)
up32 = Conv2D(32,3,activation='relu', padding='same')(up32)

conc32 = concatenate([conv32,up32], axis = 3)
up64 = UpSampling2D(size=(2,2))(conc32)
up64 = Conv2D(32,3,activation='relu', padding='same')(up64)

conc64 = concatenate([conv64,up64], axis = 3)
up128 = UpSampling2D(size=(2,2))(conc64)
up128 = Conv2D(1,3,activation='sigmoid', padding='same')(up128)

model = Model(inputs, up128)
model.compile(optimizer = optimizers.Adam(), loss= 'binary_crossentropy', metrics = ['accuracy'])

x = []
y = []
for i in range(1,175):
    train = cv2.imread('train data/train{}.jpg'.format(i))
    label = cv2.imread('labels/label{}.jpg'.format(i))
    gray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY).reshape(128,128,1)
    gray = gray/255
    x.append(train)
    y.append(gray)
x = np.array(x)
y = np.array(y)
print(x.shape, y.shape)
model.fit(x,y,epochs = 20)
model.save('unet.h5')
