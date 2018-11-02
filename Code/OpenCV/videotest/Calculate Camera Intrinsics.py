#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2 as cv
import glob
import winsound, time, os, sys


# In[2]:


# find the data file with extension .npz
outfile = glob.glob("*.npz")
outfile = outfile[0]

# load the npz file with objpoints and imgpoints array
npzfile = np.load(outfile)

# extract the two arrays...
objpoints = npzfile["objpoints"]
imgpoints = npzfile["imgpoints"]


# In[ ]:


# use this cell to modify how many images are used as reference...
ref_images = 200    # how many images to use? 

objp = objpoints[:ref_images+1]
imgp = imgpoints[:ref_images+1]

# camera calibration

startTime = time.time()

frame_width  = w = 1920
frame_height = h = 1080

# determine camera matrix, distortion coefficients, 
# rotation and translation vectors
rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera(
                                                objp, imgp, 
                                                (frame_width,frame_height), None, None)

print("\nRMS", rms, "\n")
print("Camera Matrix: \n", camera_matrix)
print("distortion coefficients: ", dist_coefs.ravel(), "\n")

# compute focal lengths
fx = camera_matrix[0][0]   # focal length in x-direction
fy = camera_matrix[1][1]   # focal length in y-direction
W  = 4                 # sensor width in mm    6.3 x 4.7mm
H  = 3                # sensor height in mm

Fx = fx * W/w
Fy = fy * H/h

print("Focal length in x = {:.2f} mm".format(Fx))
print("Focal length in y = {:.2f} mm".format(Fy))
winsound.Beep(440, 1000)
print("Processed {:.0f} images in {:.1f} s".format(ref_images, time.time()-startTime))


# In[11]:


# refine the camera matrix
newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, 
                                                dist_coefs, (w,h), 1, (w,h))

print("New Camera Matrix: \n", newcameramtx)
print("ROI = ", roi)


# In[ ]:


# undistort, need to do frame by frame and apply undistortion

