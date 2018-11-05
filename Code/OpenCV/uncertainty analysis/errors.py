import numpy as np
import cv2 as cv
import random
import matplotlib.pyplot as plt
from scipy.stats import norm, sem
import os, sys, time 

loadFile = "30000_repeats.npz"
pointFile = "point_data.npz"

def getObjpImgp(pointfile):
    """extract objectpoints and imagepoints from .npz file
    Output: objpoints, imgpoints"""
    outfile = np.load(pointfile)

    objp = outfile["objpoints"]
    imgp = outfile["imgpoints"]

    assert (len(objp) != 0 and len(imgp) != 0), "File loaded incorrectly..."

    return objp, imgp

def getParameters(loadfile):
    # extract parameters from .npz file
    outfile = np.load(loadfile)

    cxs = outfile["cxs"]            # x-principle coordinates
    cys = outfile["cys"]            # y-principle coordinates
    fxs = outfile["fxs"]            # focal lengths in x-direction
    fys = outfile["fys"]            # focal lengths in y-direction
    newcxs = outfile["newcxs"]      # optimal x-principle coordinates
    newcys = outfile["newcys"]      # optimal y-principle coordinates
    newfxs = outfile["newfxs"]      # optimal focal lengths in x
    newfys = outfile["newfys"]      # optimal focal lengths in y
    k1s = outfile["k1s"]            # first radial distortion coef
    k2s = outfile["k2s"]            # second radial distortion coef
    k3s = outfile["k3s"]            # third radial distortion coef
    p1s = outfile["p1s"]            # first tangential distortion coef
    p2s = outfile["p2s"]            # second tangential distortion coef

    return cxs, cys, newcxs, newcys, fxs, fys, newfxs, newfys, k1s, k2s, k3s, p1s, p2s

def subGroupAverages(loadfile, numSubGroups, parameter=""):
    if parameter == "":
        parameter = "fxs"

    (cxs, cys, newcxs, newcys, fxs, fys, 
    newfxs, newfys, k1s, k2s, k3s, p1s, p2s) = getParameters(loadfile)

    parameterDict = {"cxs":cxs, "cys":cys, "newcxs":newcxs, "newcys":newcys,
    "fxs":fxs, "fys":fys, "newfxs":newfxs, "newfys":newfys, "k1s":k1s,
    "k2s":k2s, "k3s":k3s, "p1s":p1s, "p2s":p2s,}

    # find the specific parameter data
    data = parameterDict[parameter]

    # split the data into various subgroups
    splitData = np.array_split(data, numSubGroups)

    # compute the average value for each sub-group
    averages = np.zeros(numSubGroups)
    for i in range(numSubGroups):
        averages[i] = np.mean(splitData[i])

    stdError = sem(averages)

    # plot the data
    plt.figure()


    mu, std = norm.fit(averages)
    # remove outliers if further than 3 std from mean
    averages = [x for x in averages if ((x > mu - 3*std) and (x < mu + 3*std))]
    mu, std = norm.fit(averages)

    plt.hist(averages, bins=100, density=True, edgecolor="k")


    print("mean is {}".format(mu))
    print("standard deviation is {}".format(std))
    print("Standard Error is {}\n".format(stdError))
    xmin, xmax = plt.xlim()
    print("xmin, xmax = ", xmin, xmax)

    x = np.linspace(xmin, xmax, 1000)
    y = norm.pdf(x, mu, std)
    
    plt.plot(x,y, linewidth=2)

    plt.ylabel(r"Probability Density")
    plt.xlabel(parameter)
    

    plt.show()


    None
 
#subGroupAverages(loadFile, 2000, "newcys")

# reference paramaters computed from 150 randomly selected reference images
# ... see calculate camera intrinsics jupyter notebook 
# ... in videotest folder
referenceDict = {
"ref_fx" : 1.87328075e+03,
"ref_fy" : 1.87411535e+03,
"ref_cx" : 9.68886059e+02,
"ref_cy" : 5.49519121e+02,
"ref_k1" :  1.20516762e-01,
"ref_k2" : 1.18944527e-01,
"ref_k3" : -1.34833637e+00,
"ref_p1" : 2.52464884e-03,
"ref_p2" : -4.75592880e-04,
"ref_opt_fx" : 1.86776758e+03,
"ref_opt_fy" : 1.87207007e+03,
"ref_opt_cx" : 9.69674507e+02,
"ref_opt_cy" : 5.52861470e+02,
}

def getPoints(pointfile, groupSize):
    """Returns specified number of radomly selected objpoints 
    and imgpoints from .npz file. May need optimising..."""
    objpoints, imgpoints = getObjpImgp(pointfile)

    objp, imgp = [], []
    indexes = []

    while len(indexes) < groupSize:
        index = random.randint(0, len(objpoints)-1)
        if index in indexes:
            continue
        else:
            indexes.append(index)
            objp.append(objpoints[index])
            imgp.append(imgpoints[index])


    return objp, imgp

def generateMTX(pointfile, groupSize, image_width, image_height):
    """Generate a camera matrix from a set group size of images
    in a sample of images"""
    objp, imgp = getPoints(pointfile, groupSize)
    w, h = image_width, image_height

    # compute camera intrinsics
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera(
                                                    objp, imgp, (w,h),
                                                    None, None)
    # refine camera matrix
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix,
                                                     dist_coefs, (w,h),
                                                     1, (w,h))

    return camera_matrix, newcameramtx, dist_coefs, rvecs, tvecs

def getCameraIntrinsics(cameraMatrix, optimalCameraMatrix):
    """From camera matrix and optimal Camera Matrix return fx, fy
    cx, cy, opt_fx, opt_fy, opt_cx, opt_cy"""
    fx = cameraMatrix[0][0]
    fy = cameraMatrix[1][1]
    cx = cameraMatrix[0][2]
    cy = cameraMatrix[1][2]
    opt_fx = optimalCameraMatrix[0][0]
    opt_fy = optimalCameraMatrix[1][1]
    opt_cx = optimalCameraMatrix[0][2]
    opt_cy = optimalCameraMatrix[1][2]

    return fx, fy, cx, cy, opt_fx, opt_fy, opt_cx, opt_cy

def getDistortionCoefs(distCoefs):
    """Extract the distortion coefficients from list of 
    distortion coefficients"""
    distortionCoefs = distCoefs.ravel()
    k1 = distortionCoefs[0]
    k2 = distortionCoefs[1]
    p1 = distortionCoefs[2]
    p2 = distortionCoefs[3]
    k3 = distortionCoefs[4]

    return k1, k2, p1, p2, k3 


def compareSample(pointfile, groupSize, repetitions,  image_width, image_height, parameter=""):
    if parameter == "":
        parameter = "cx"

    # extract object points and image points from pointfile
    objp, imgp = getObjpImgp(pointfile)    

    # get the reference parameter value from dictionary
    refParameter = referenceDict["ref_"+parameter]

    # array to store parameter values over various repetitions
    xs = np.zeros(repetitions)

    for i in range(repetitions):
        if i%3 == 0:
            dots = "..."
        elif i%2 == 0: 
            dots = ".."
        else:
            dots = "."
        sys.stdout.write("\r"+"{:.1f} % Complete{}".format(i/repetitions * 100, dots))
        camera_matrix, newcameramtx, dist_coefs, rvecs, tvecs = generateMTX(pointfile, groupSize, 
                                                                            image_width, image_height)

        # extract camera parameters from matrices ^^                               
        fx, fy, cx, cy, opt_fx, opt_fy, opt_cx, opt_cy = getCameraIntrinsics(camera_matrix, newcameramtx)
        k1, k2, p1, p2, k3 = getDistortionCoefs(dist_coefs)

        # dictionary of parameters
        parameterDict = {"fx":fx, "fy":fy, "cx":cx, "cy":cy, "newfx":opt_fx, "newfy":opt_fy, 
                        "newcx": opt_cx, "newcy":opt_cy, "k1":k1, "k2":k2, "k3":k3, "p1":p1, 
                        "p2":p2,}

        xs[i] = parameterDict[parameter]

    mu, std = norm.fit(xs)
    # remove outliers
    newxs = [x for x in xs if ((x > mu - 3*std) and (x < mu + 3*std))]
    mu, std = norm.fit(newxs)

    plt.figure()

    # plot histogram of parameter
    plt.hist(newxs, bins=70, density=True, edgecolor="k")
    ymin, ymax = plt.ylim()
    plt.vlines(refParameter, 0, ymax, "r", linewidth=2)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 1000)
    y = norm.pdf(x, mu, std)

    plt.plot(x, y, "o-")

    plt.show()

compareSample(pointFile, 10, 1000, 1920, 1080, "cx")
            





    
    











    





