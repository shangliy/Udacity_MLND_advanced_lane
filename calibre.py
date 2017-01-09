#!usr/bin/python
'''
author: Shanglin Yang (kudoysl@gmail.com)

This script works to compute calibration matrix based on 
the images for camera calibration stored in the folder called `camera_cal`.
And save the amera calibration matrix ['mtx'] and distortion coefficients ['dist'] to a pickle file 'wide_dist_pickle.p'
'''
import sys
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


row = 9  #The row of points for calibration
col = 6  #The column of points for calibration
objp = np.zeros((row*col,3),np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    print(idx)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    print(ret)
    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(img, (9,6), corners, ret)
        # If found, draw corners
        plt.imshow(img)
        plt.show()
        

cv2.destroyAllWindows()

import pickle

# Test undistortion on an image
img = cv2.imread('camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)


dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('camera_cal/test_undist.jpg',dst)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "./wide_dist_pickle.p", "wb" ) )
#dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)
plt.show()

# Make a list of test images
test_images = glob.glob('test_images/*.jpg')

for idx, fname in enumerate(test_images):
    print(fname)
    img = cv2.imread(fname)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst_show = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img_show)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst_show)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.show()



