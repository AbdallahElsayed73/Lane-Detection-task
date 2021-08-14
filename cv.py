import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import sys
from scipy import stats

# first we apply canny on all the images to detect the sharp edges in it

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

# then we need to focus on some region in front of the car to detect the lane not the whole image

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    
    if len(img.shape) > 2:
    # this is where the image contains more than one channel like an RGB picture
        channel_count = img.shape[2]  
        ignore_mask_color = (255,) * channel_count
    else:
    # for gray-scale images
        ignore_mask_color = 255

    # filling a polygon with ones and zeroes using the vertices we have decided to and then anding it with the image
    # to get a masked image with only the region we want  
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
# then we apply a hough transform to detect the lines in the region of interest which should define the lane lines
def hough_transform(canny_img, rho, theta, threshold, min_line_len, max_line_gap):
    return cv2.HoughLinesP(canny_img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

# this method is to draw 2 lines given its start and end points and fill the area between them which should represent the lane
def draw_lines(img, left_lines,right_lines, color=[255, 0, 0], thickness=5, make_copy=True):
    img_copy = np.copy(img) if make_copy else img
    ignore_mask_color= (255,0,0)
    
    vertices = np.array([[left_lines[0], left_lines[1], right_lines[1], right_lines[0]]], dtype=np.int32)
    cv2.fillPoly(img_copy, vertices, ignore_mask_color)
    return img_copy
    
# this method seperates the left lines from the right lines using their slope and coordinates    
def separate_lines(lines, img):
    img_shape = img.shape
    
    middle_x = img_shape[1] / 2
    
    left_lane_lines = []
    right_lane_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:

            # vertical and horizontal lines are discarded we are only interested with the lines with some slope that are either 
            # sloping to the left or to the right
            dx = x2 - x1 
            if dx == 0:
                #Discarding line since we can't gradient is undefined at this dx
                continue
            dy = y2 - y1
            
            # Similarly, if the y value remains constant as x increases, discard line
            if dy == 0:
                continue
            
            slope = dy / dx
            
            # This is pure guess than anything... 
            # but get rid of lines with a small slope as they are likely to be horizontal one
            epsilon = 0.1
            if abs(slope) <= epsilon:
                continue
            
            if slope < 0 and x1 < middle_x and x2 < middle_x:
                # Lane should also be within the left hand side of region of interest
                left_lane_lines.append([[x1, y1, x2, y2]])
            elif x1 >= middle_x and x2 >= middle_x:
                # Lane should also be within the right hand side of region of interest
                right_lane_lines.append([[x1, y1, x2, y2]])
    
    return left_lane_lines, right_lane_lines

# this method combines different lines into one line using the linregress method that returnes the slope and the intercept with
# the y axis for that line  
def find_lane_lines_formula(lines):
    xs = []
    ys = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            xs.append(x1)
            xs.append(x2)
            ys.append(y1)
            ys.append(y2)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(xs, ys)
    
    # Remember, a straight line is expressed as f(x) = Ax + b. Slope is the A, while intercept is the b
    return (slope, intercept)


# this method just take the lines that should be regressed into one line and returns the coordinates of the 2 points representing
# the line equivalent to them
def trace_lane_line(img, lines, top_y, make_copy=True):
    A, b = find_lane_lines_formula(lines)
    lowerLeftPoint = [0, 600]
    upperLeftPoint = [600, 310]
    upperRightPoint = [680, 310]
    lowerRightPoint = [1170, 720]

    pts = np.array([[lowerLeftPoint, upperLeftPoint, upperRightPoint, 
    lowerRightPoint]], dtype=np.int32)

    img_shape = img.shape
    bottom_y = img_shape[0] - 1
    # y = Ax + b, therefore x = (y - b) / A
    x_to_bottom_y = (bottom_y - b) / A
    
    top_x_to_y = (top_y - b) / A 
    
    new_lines = [[int(x_to_bottom_y), int(bottom_y)], [int(top_x_to_y), int(top_y)]]
    return new_lines


# this method here takes the lines defining the left and right of the lane, regress both of them into 2 lines that represent the borders
# of the lane and then draw these lines on the image calling the draw_lines method
def trace_both_lane_lines(img, left_lane_lines, right_lane_lines):
    lowerLeftPoint = [0, 600]
    upperLeftPoint = [600, 310]
    upperRightPoint = [680, 310]
    lowerRightPoint = [1170, 720]

    pts = np.array([[lowerLeftPoint, upperLeftPoint, upperRightPoint, 
    lowerRightPoint]], dtype=np.int32)
    region_top_left = pts[0][1]
    
    full_left_lane_lines = trace_lane_line(img, left_lane_lines, region_top_left[1], make_copy=True)
    full_left_right_lanes_lines = trace_lane_line(img, right_lane_lines, region_top_left[1], make_copy=False)
    
    
    
    return draw_lines(img,full_left_lane_lines,full_left_right_lanes_lines)



# now, it's time to pipeline it all
def pipeline(image):
  # we will first start with converting the image to gray scale which makes it easier to apply hough transform on
  # then applying a gaussianBlur filter to reduce the noise in the image  
  grayscaled= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gaussianBlur = cv2.GaussianBlur(grayscaled, (5, 5), 0)

  # then applying the canny method on the resulting image with a min and max threshole of 50 and 150 that were deduced experimentally
  minThreshold = 50
  maxThreshold = 150
  edgeDetectedImage = canny(gaussianBlur, minThreshold, maxThreshold)
  # this is our region of interest that is only convenient for this video
  lowerLeftPoint = [0, 600]
  upperLeftPoint = [600, 310]
  upperRightPoint = [680, 310]
  lowerRightPoint = [1170, 720]

  pts = np.array([[lowerLeftPoint, upperLeftPoint, upperRightPoint, 
  lowerRightPoint]], dtype=np.int32)
  masked_image = region_of_interest(edgeDetectedImage, pts)

  # values for rho, theta, the thresholdn minimum line length, and maximum line gap were all deduced experimentally using different combinations

  rho = 1
  theta = (np.pi/180) * 1
  threshold = 20
  min_line_length = 15
  max_line_gap = 15
  # applying hough transform on the masked image to get all the lines in it
  houged = hough_transform(masked_image, rho, theta, 
                    threshold, min_line_length, max_line_gap)
  # then seperating these lines using the seperate_line method which returns 
  separated_lanes = separate_lines(houged, image)

  # and the resulting image is gonna be the output of trace_both_lane_lines by giving it the lines representing the left and right borders
  # then resizing it to 128*128 which we will use in the deep learning model
  res = trace_both_lane_lines(image, separated_lanes[0], separated_lanes[1])
  return cv2.resize(res,(128,128))


video = cv2.VideoCapture('Lane detect test data.mp4')
i=0
count=1
# to overcome repeating the same images we will one taje one frame out of 10 because nothing much will change between 2 consecutive frames
while ret:
    ret,frame = video.read()
    if ret:
        i+=1
    if(i%10==1):
        # we have 2 folders one for the images to be trained on which is simply taking the frame from the video and resizing it
        # and the other of the results from the CV after applying the pipeline on that frame
        scaled = cv2.resize(frame, (128,128))
        cv2.imwrite('train data/train{}.jpg'.format(count), scaled)
        res = pipeline(frame)
        cv2.imwrite('results cv/res{}.jpg'.format(count), res)
        count+=1




