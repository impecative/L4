#!/usr/bin/env python
# coding: utf-8

# In[1]:


# necessary modules
import numpy as np      # for maths
import cv2 as cv        # for camera tools
import glob             
import os
import sys, time


# In[2]:


# termination criteria
criteria = (cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER, 30, 0.001)

cols, rows = 7, 10

shape = (cols, rows)


# In[3]:


# prepare object points
# prepare object points
objp = np.zeros((cols*rows,3), np.float32) # zero array for 8 x 11 circle board
objp[:,:2] = np.mgrid[0:cols,0:rows].T.reshape(-1,2)  # format shape of array

# arrays to store object points and image points
objpoints = []
imgpoints = []


# In[4]:


# read the image file(s)
images = glob.glob("*.JPG")

folder = "found_patterns"

# if the folder doesn't exist, make it
if not os.path.exists(folder):
    os.mkdir(folder)

# folder file path    
path = "{}/{}".format(os.getcwd(), folder)

counter, success = 1, 0 
size = (cols,rows)   # (cols, rows)
startTime = time.time()

path = '{}/{}'.format(os.getcwd(),folder)

for fname in images:
       
    # progress counter
    percent_done = counter * 100 / len(images)
    sys.stdout.write('\r' + "Image {:.0f} of {:.0f}. Time elapsed: {:.0f} s".format(counter, len(images), time.time()-startTime))
    counter += 1
    
    # full size image for best accuracy
    img = cv.imread(fname)
    
    # resize the image to make it more manageable
    #reimg = cv.resize(img, (800, 600))    # (1149, 766) works
    reimg = img
    
    # scale factor
    factor = img.shape[1]/reimg.shape[1]

    # convert to gray
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # find checkerboard corners
    ret, centres = cv.findChessboardCorners(gray, size, 
                                        flags=cv.CALIB_CB_ADAPTIVE_THRESH
                                           + cv.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True: 
        success += 1
        sys.stdout.write("\r"+"Pattern found... Time elapsed {:.0f}".format(
                                                                    time.time()-startTime))
        objpoints.append(objp)
    
        centres2 = centres
        centres2 = cv.cornerSubPix(gray, centres, (11,11), (-1,-1), 
                                   criteria)

        imgpoints.append(centres2)

        # draw and display the patterns
        drawimg = cv.drawChessboardCorners(reimg, size, centres2/factor, ret)

        #cv.imshow("img", drawimg)        
        cv.imwrite(os.path.join(path , '{}.png'.format(success)), drawimg)

        cv.waitKey(200)
        
    else:
        sys.stdout.write("\r" + "{:.0f} %, Pattern not found...".format(percent_done))
        
cv.destroyAllWindows()
        
sys.stdout.write("\r" +"Succeeded for {}/{} images".format(success,len(images)))

outfile = "data_points.npz"

np.savez(outfile, objpoints=objpoints, imgpoints=imgpoints)



