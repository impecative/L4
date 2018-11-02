import numpy as np
import cv2 as cv
import glob
import random
import matplotlib.pyplot as plt
import winsound, time, os, sys

# find the data file with extension .npz
outfile = "data_points.npz"

# load the npz file with objpoints and imgpoints array
npzfile = np.load(outfile)

# extract the two arrays...
objpoints = npzfile["objpoints"]
imgpoints = npzfile["imgpoints"]

def pickRandom(number, objpoints, imgpoints):
    """Pick random indexes of images in sample"""
    indexs = []
    # pick random index for extracting 
    for i in range(number):
        index = random.randint(0, len(objpoints)-1)
        if not index in indexs:
            indexs.append(index)
        else:
            indexs.append(random.randint(0, len(objpoints)-1)) 


    return indexs

def generateMTX(number_images, objpoints, imgpoints):
    """Generate a camera matrix from a set number of images
    in a sample of images"""
    w, h = 2992, 2000

    objp = []
    imgp = []

    # pick the random indexes and extract
    # objpoints, imgpoints
    indexs = pickRandom(number_images, objpoints, imgpoints)
    #print(indexs)
    for i in indexs:
        objp.append(objpoints[i])
        imgp.append(imgpoints[i])


    # compute camera intrinsics
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera(
                                                    objp, imgp, (w,h),
                                                    None, None)
    # refine camera matrix
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix,
                                                     dist_coefs, (w,h),
                                                     1, (w,h))

    return camera_matrix, newcameramtx, dist_coefs, rvecs, tvecs

def getFocalLength(camera_matrix):
    fx = camera_matrix[0][0]
    fy = camera_matrix[1][1]

    return fx, fy

def getPrincipalPoints(camera_matrix):
    cx = camera_matrix[0][2]
    cy = camera_matrix[1][2]

    return cx, cy

def getDistCoefs(dist_coefs):
    distortionCoefs = dist_coefs.ravel()
    k1 = distortionCoefs[0]
    k2 = distortionCoefs[1]
    p1 = distortionCoefs[2]
    p2 = distortionCoefs[3]
    k3 = distortionCoefs[4]

    return k1, k2, p1, p2, k3

def main():
    startTime = time.time()

    repetitions = 500

    # store focal lengths 
    fxs    = np.zeros(repetitions)    # use np array for speed! 
    fys    = np.zeros(repetitions)
    newfxs = np.zeros(repetitions)
    newfys = np.zeros(repetitions)
    cxs    = np.zeros(repetitions)
    cys    = np.zeros(repetitions)
    newcxs = np.zeros(repetitions)
    newcys = np.zeros(repetitions)
    k1s    = np.zeros(repetitions)
    k2s    = np.zeros(repetitions) 
    k3s    = np.zeros(repetitions)
    p1s    = np.zeros(repetitions)
    p2s    = np.zeros(repetitions)

    # fill the arrays! 
    counter = 1
    for i in range(repetitions):
        dots = None
        if counter%4 == 0:
            dots = ""
        elif counter%3 == 0:
            dots = "..."
        elif counter%2 == 0:
            dots = ".."
        else:
            dots = "."
       
        sys.stdout.write("\r"+"{:.1f} % complete".format(i/repetitions *100) + dots)
        camera_matrix, newcameramtx, dist_coefs = generateMTX(10, objpoints, imgpoints)[:3]
        fx, fy = getFocalLength(camera_matrix)
        cx, cy = getPrincipalPoints(camera_matrix)
        newcx, newcy = getPrincipalPoints(newcameramtx)
        newfx, newfy = getFocalLength(newcameramtx)
        k1, k2, p1, p2, k3 = getDistCoefs(dist_coefs)

        fxs[i]    = fx
        fys[i]    = fy
        newfxs[i] = newfx
        newfys[i] = newfy
        cxs[i]    = cx
        cys[i]    = cy
        newcxs[i] = newcx
        newcys[i] = newcy
        k1s[i]    = k1
        k2s[i]    = k2
        k3s[i]    = k3
        p1s[i]    = p1
        p2s[i]    = p2

        counter += 1

    #### save the imgpoints and objpoints ####
    #### for analysis in another module ####
    savefile = "uncertainty_data_{}_10_images.npz".format(repetitions)

    np.savez(savefile, fxs=fxs, fys=fys, newfxs=newfxs, newfys=newfys,
            cxs=cxs, cys=cys, newcxs=newcxs, newcys=newcys, k1s=k1s,
            k2s=k2s, k3s=k3s, p1s=p1s, p2s=p2s)

    print("\n" + "Completed in {:.1f} mins".format((time.time()-startTime)/60))

    None

if __name__ == "__main__":
    main()


    
    
        
    
