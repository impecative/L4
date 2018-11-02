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
images = glob.glob("*.jpg")

folder = "found_patterns"

if not os.path.exists(folder):
    os.mkdir(folder)
    
path = "{}/{}".format(os.getcwd(), folder)

# In[ ]:


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
    reimg = cv.resize(img, (800, 600))    # (1149, 766) works
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


# In[ ]:


# camera calibration
h, w = img.shape[:2]

rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera(
                                                objpoints, imgpoints, 
                                                (w,h), None, None)

print("\nRMS", rms, "\n")
print("Camera Matrix: \n", camera_matrix)
print("distortion coefficients: ", dist_coefs.ravel(), "\n")

# compute focal lengths
fx = camera_matrix[0][0]    # focal length in x-direction
fy = camera_matrix[1][1]    # focal length in y-direction
W  = 4                      # sensor width in mm
H  = 3                      # sensor height in mm

Fx = fx * W/w
Fy = fy * H/h

print("Focal length in x = {:.2f} mm".format(Fx))
print("Focal length in y = {:.2f} mm".format(Fy))


# In[ ]:


newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, 
                                                dist_coefs, (w,h), 1, (w,h))

print("New Camera Matrix: \n", newcameramtx)
print("ROI = ", roi)


# In[ ]:


# check the camera parameters
fovx, fovy, focalLength, principalPoint, aspectRatio = cv.calibrationMatrixValues(camera_matrix, img.shape[:2], 
                                                                                  15.6, 23.5)

print("focal length is {:.2f} mm \n".format(focalLength))
print("aspect ratio is {:.2f} \n".format(aspectRatio))
print("The principle point is ({:.2f}, {:.2f})".format(principalPoint[0], principalPoint[1]))


# In[ ]:


# undistort the image

dst = cv.undistort(reimg, camera_matrix, dist_coefs, None, newcameramtx)


# In[ ]:


# crop and display image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]


# In[ ]:


# show before and after distortion
#resize = dst.resize(dst, (600, 400))
#cv.imshow("the result", resize)
#cv.namedWindow("The result!", flags=WINDOW_AUTOSIZE)

cv.imshow("The result", dst)
cv.imshow("original", reimg)

new = cv.resize(dst, (600, 400))
original = cv.resize(reimg, (600,400))
#cv.imshow("The result!", new)



cv.imshow("Original", original)
cv.imshow("New!", new)

cv.waitKey(0)

cv.destroyAllWindows()


# In[ ]:


# calculate rms error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coefs)
    error = cv.norm(imgpoints[i],imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print("total error: ", mean_error/len(objpoints))


# In[ ]:





# In[ ]:




