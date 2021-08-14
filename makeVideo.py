from operator import truediv
from os import write
import numpy as np
import sklearn
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

# this algorithm loads the video we trained on earlier and loads the trained model and then predicts every frame in that video
# and make a resulting video with avi extension
model = tf.keras.models.load_model('unet.h5')

video = cv2.VideoCapture('Lane detect test data.mp4')
results = []



ret = True
i=1
while ret:
    ret, frame = video.read()
    if ret:
        # for every frame we will make the model predict every pixel in it and decide whether it's a part of the lane or not
        pred = model.predict(cv2.resize(frame,(128,128)).reshape(1,128,128,3))

        # our threshold is 50% so if the model gives a propability that a pixel is more than 50% a lane it's considered a lane
        pred[pred>=0.5] =1
        pred[pred<0.5] = 0
        # then multiplying by 255 to return it to a black and white image
        pred = (pred*255).astype(np.uint8)
        results.append(pred.reshape(128,128))
        print("frame {}".format(i))
        i+=1

# then taking all the predicted frames and we will write a video using the cv2.VideoWriter class and that's it
fps = video.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('result_deep.avi', fourcc, fps, (128,128))
for f in results: 
    tmp = np.zeros((128,128,3)).astype(np.uint8)
    tmp[:,:,0]= f
    tmp[:,:,1]= f
    tmp[:,:,2]= f
    out.write(tmp)
cv2.destroyAllWindows()
out.release()
