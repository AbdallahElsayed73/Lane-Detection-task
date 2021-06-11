from operator import truediv
from os import write
import numpy as np
import sklearn
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

model = tf.keras.models.load_model('unet.h5')

video = cv2.VideoCapture('Lane detect test data.mp4')
results = []
# ret, frame = video.read()
# pred = model.predict(np.array([cv2.resize(frame,(128,128))]))[0]
# pred[pred>=0.6] = 255
# pred[pred<0.6]=0
# tmp = np.zeros((128,128,3))
# for i in range(128):
#     for j in range(128):
#         tmp[i][j][0]= pred[i][j]
#         tmp[i][j][1]= pred[i][j]
#         tmp[i][j][2]= pred[i][j]
# cv2.imshow('image',tmp)
# cv2.waitKey(0)  


ret = True
i=1
while ret:
    ret, frame = video.read()
    if ret:
        pred = model.predict(cv2.resize(frame,(128,128)).reshape(1,128,128,3))
        pred[pred>=0.5] =1
        pred[pred<0.5] = 0
        pred = (pred*255).astype(np.uint8)
        results.append(pred.reshape(128,128))
        print("frame {}".format(i))
        i+=1

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
