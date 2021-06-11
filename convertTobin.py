import cv2
import numpy as np
import matplotlib.pyplot as plt

for i in range(1,175): 
    res = np.zeros((128,128))
    frame = cv2.imread('results cv/res{}.jpg'.format(i))
    for x in range(128):
        for y in range(128):
            color = frame[x,y]
            if(color[0]>240 and color[1]<15 and color[2]<15):
                res[x,y] = 255
    cv2.imwrite('labels/label{}.jpg'.format(i),res)

frame = cv2.imread('results cv/res1.jpg')
print(frame[100,60])
plt.imshow(frame)
plt.show()