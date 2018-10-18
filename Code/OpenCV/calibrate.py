
from __future__ import print_function # python 2/3 compatibility
import numpy as np
import cv2 
import glob

# termination criteria
criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0)
objp = np.zeros((6*7,3), np.float32) # zero array for 6x7 chessboard
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane. 

images = glob.glob("*.jpg")

img = cv2.imread(images[2])

print(images)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6))

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7,6), corners, ret)
        cv2.imshow("img", img)
        cv2.waitKey(100)

cv2.destroyAllWindows()


# calibration
# cv2.calibrateCamera() returns the camera matrix, distortion coefficients, rotation and translation vectors...
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# # undistortion

# # refine camera matrix...
# img = cv2.imread('left12.jpg')
# h,  w = img.shape[:2]
# newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
# # cv2.imshow("roi", roi)
# # cv2.waitKey(1000)

# # undistort
# mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
# dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

# # crop the image
# x,y,w,h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.imwrite('calibresult.png',dst)
# cv2.imshow("This is the result!", dst)
# cv2.waitKey(2000)
# cv2.imwrite("result.png", dst)

# cv2.destroyAllWindows(
    
h,  w = img.shape[:2]

rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w,h), None, None)

print("Image (w,h) = ", (w,h))
print("\nRMS", rms)
print("camera matrix: \n", camera_matrix)
print("distortion coefficients: ", dist_coefs.ravel())

# undistort image with the calibration
print("")

print("Using image: ", images[2])

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w,h), 1, (w,h))

dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

print("New Camera Matrix: \n", newcameramtx)
print("ROI = ", roi)

# crop and display image
x, y, w, h = roi 
dst = dst[y:h, x:w]

cv2.imwrite("result.png", dst)
cv2.imshow("the result! ", dst)
cv2.waitKey(0)

cv2.destroyAllWindows()


