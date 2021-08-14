import numpy as np
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras import backend as keras
from keras import optimizers
import cv2 

# the model is based on the famous U-net architecture which is a series of downsampling followed by upsampling until we reach
# the original size of the image

# our model is gonna start with a series of 1 convolutional layer followed be a downsampling layer using a maxpool until the image reaches 8*8 pixels
# the input is our original 128*128 RGB image
# all the conv layers contain 32 filters with a relu activation function
# they all have a padding value of same in order for the maxpool to reduce its size to a power of 2

inputs = Input((128,128,3))

# the first layer
conv128 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
pool128 = MaxPool2D(pool_size=(2,2))(conv128)

# the second layer
conv64 = Conv2D(32,3,activation='relu', padding='same')(pool128)
pool64 = MaxPool2D(pool_size=(2,2))(conv64)

# the third layer
conv32 = Conv2D(32,3,activation='relu', padding='same')(pool64)
pool32 = MaxPool2D(pool_size=(2,2))(conv32)

# the fourth layer
conv16 = Conv2D(32,3,activation='relu', padding='same')(pool32)
pool16 = MaxPool2D(pool_size=(2,2))(conv16)

# then at the bottom of the architecture we have a series of 2 convolutional layers
conv8 = Conv2D(32,3,activation='relu', padding='same')(pool16)
conv8 = Conv2D(32,3,activation='relu', padding='same')(conv8)

# now for the upsamling it's gonna be a series of upsampling filters followed by 1 conv layer

# we are concatenation the result of the upsampling with its counter result from the downsampling to get better information about
# the image when reconstructing it the upper level
# we are using a concatenate layer to concatenate the information coming from the upsampling and th downsamping

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
# the last convolutional layer has to have a sigmoid activation function because this is a binary classification problem
up128 = Conv2D(1,3,activation='sigmoid', padding='same')(up128)

# we will use Adam optimizer along with binary crossentropy loss function which is convenient for this binary classification problem
model = Model(inputs, up128)
model.compile(optimizer = optimizers.Adam(), loss= 'binary_crossentropy', metrics = ['accuracy'])


# our data is saved in the train and labels folder we constructed earlier using the CV algorithm
X = []
y = []
for i in range(1,175):
    train = cv2.imread('train data/train{}.jpg'.format(i))
    label = cv2.imread('labels/label{}.jpg'.format(i))
    gray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY).reshape(128,128,1)
    gray = gray/255
    X.append(train)
    y.append(gray)
x = np.array(X)
y = np.array(y)
print(X.shape, y.shape)
# we train the model for only 20 epochs which takes a little bit of time but it gives sime good results
model.fit(X,y,epochs = 20)

# and finally saving the model with the h5 extension to using it when predicting later
model.save('unet.h5')
