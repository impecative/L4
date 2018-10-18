
from __future__ import print_function # python 2/3 compatibility
import numpy as np
import cv2 
import glob

# termination criteria
criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0)
objp = np.zeros((8*11,3), np.float32) # zero array for 6x7 chessboard
objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

# Arrays to store object points and image points from all images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane. 

images = glob.glob("*.jpg")

if not len(images) == 0:
    img = cv2.imread(images[0])

print(images)

# need at least 10 patterns for camera callibration...

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (11,8))

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners  (uncomment to draw on corners)
        cv2.drawChessboardCorners(img, (11,8), corners2, ret)
        cv2.imshow("img", img)   
        cv2.waitKey(100)

cv2.destroyAllWindows()


# calibration
# cv2.calibrateCamera() returns the camera matrix, distortion coefficients, rotation and translation vectors...
    
h,  w = img.shape[:2]

rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w,h), None, None)

print("Image (w,h) = ", (w,h))
print("\nRMS", rms)
print("camera matrix: \n", camera_matrix)
print("distortion coefficients: ", dist_coefs.ravel())

# undistort image with the calibration
print("")

print("Using image: ", images[0])

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w,h), 1, (w,h))

dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

print("New Camera Matrix: \n", newcameramtx)
print("ROI = ", roi)

# crop and display image
x, y, w, h = roi 
dst = dst[y:h+y, x:w+x]

# check accuracy of result
mean_error = 0
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coefs)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print("total error: ", mean_error/len(objpoints))



cv2.imwrite("result.png", dst)
cv2.imshow("the result! ", dst)
cv2.waitKey(0)

cv2.destroyAllWindows()



