from __future__ import division    # compatibility with python 2
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
    # print("mean, std = ", mu, std)
    # remove outliers if further than 3 std from mean
    averages2 = [x for x in averages if ((x > mu - 3*std) and (x < mu + 3*std))]
    # print("The lowest x value is ", np.min(averages2))
    # print("The highest x value is ", np.max(averages2))

    mu, std = norm.fit(averages2)

    plt.hist(averages2, bins=70, density=True, edgecolor="k")
    _, ymax = plt.ylim()
    xmin, xmax = plt.xlim()

    print("mean is {}".format(mu))
    print("standard deviation is {}".format(std))
    print("Standard Error is {}\n".format(stdError))
    print("xmin, xmax = ", xmin, xmax)

    x = np.linspace(xmin, xmax, 1000)
    y = norm.pdf(x, mu, std)
    
    plt.plot(x, y, linewidth=2, label="Gaussian Distribution")
    plt.vlines(mu, 0, ymax, "g", linewidth=2, label="Mean={:.1g}".format(mu))

    plt.ylabel("Probability Density")
    plt.xlabel(parameter)
    plt.legend()

    # plt.savefig("method_1_500_subgroups/{}_graph.png".format(parameter))
    plt.show()


    None
 


# reference paramaters computed from 150 randomly selected reference images
# ... see calculate camera intrinsics jupyter notebook 
# ... in videotest folder
# Process took ~33 minutes! 
referenceDict = {
"ref_fx" : 1.87275285e+03,
"ref_fy" : 1.87696998e+03,
"ref_cx" : 9.75764866e+02,
"ref_cy" : 5.42590101e+02,
"ref_k1" :   1.58175072e-01,
"ref_k2" : -6.59545156e-01,
"ref_k3" :  1.87636309e+00,
"ref_p1" : 1.10951578e-03,
"ref_p2" : -7.72102192e-04,
"ref_opt_fx" : 1.92431360e+03,
"ref_opt_fy" : 1.89481995e+03,
"ref_opt_cx" : 9.73515460e+02,
"ref_opt_cy" : 5.43054368e+02,
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


def compareSample(pointfile, groupSize, repetitions,  image_width, image_height, indexes):
    # extract object points and image points from pointfile
    objpoints, imgpoints = getObjpImgp(pointfile)

    # use the correct img points, use only indexes used for ref_parameters
    objp, imgp = [], []
    for i in range(len(indexes)):
        objp.append(objpoints[indexes[i]])
        imgp.append(imgpoints[indexes[i]])    

    # array to store parameter values over various repetitions
    fxs = np.zeros(repetitions)
    fys = np.zeros(repetitions)
    newfxs = np.zeros(repetitions)
    newfys = np.zeros(repetitions)
    cxs = np.zeros(repetitions)
    cys = np.zeros(repetitions)
    newcxs = np.zeros(repetitions)
    newcys = np.zeros(repetitions)
    k1s = np.zeros(repetitions)
    k2s = np.zeros(repetitions)
    k3s = np.zeros(repetitions)
    p1s = np.zeros(repetitions)
    p2s = np.zeros(repetitions)

    # need to output all the parameters, and save to file to make
    # analysis more efficient, rather than running this function 
    # separately for each parameter...

    for i in range(repetitions):
        sys.stdout.write("\r"+"{:.1f} % Complete".format(i/repetitions * 100))
        camera_matrix, newcameramtx, dist_coefs, rvecs, tvecs = generateMTX(pointfile, groupSize, 
                                                                            image_width, image_height)

        # extract camera parameters from matrices ^^                               
        fxs[i], fys[i], cxs[i], cys[i], newfxs[i], newfys[i], newcxs[i], newcys[i] = getCameraIntrinsics(camera_matrix, newcameramtx)
        k1s[i], k2s[i], p1s[i], p2s[i], k3s[i] = getDistortionCoefs(dist_coefs)

    # save the parameter arrays for future use...
    savefile = "parameters_method_2_({}_repetitions).npz".format(repetitions)
    np.savez(savefile, fxs=fxs, fys=fys, newfxs=newfxs, newfys=newfys,
            cxs=cxs, cys=cys, newcxs=newcxs, newcys=newcys, k1s=k1s,
            k2s=k2s, k3s=k3s, p1s=p1s, p2s=p2s)


    



def Method2(savefile, parameter=""):
    if parameter == "":
        parameter = "cx"

    # get the reference parameter value from dictionary
    if (parameter == "newcx" or parameter == "newfx" or parameter == "newfy" or parameter == "newcy"):
        refParameter = referenceDict["ref_opt_"+parameter[3:]]
    else:
        refParameter = referenceDict["ref_"+parameter]

    npzfile = np.load(savefile)
    
    # extract the arrays
    cxs = npzfile["cxs"]
    cys = npzfile["cys"]
    newcxs = npzfile["newcxs"]
    newcys = npzfile["newcys"]
    fxs = npzfile["fxs"]
    fys = npzfile["fys"]
    newfxs = npzfile["newfxs"]
    newfys = npzfile["newfys"]
    k1s = npzfile["k1s"]
    k2s = npzfile["k2s"]
    k3s = npzfile["k3s"]
    p1s = npzfile["p1s"]
    p2s = npzfile["p2s"]


    parameterDict = {"fx":fxs, "fy":fys, "cx":cxs, "cy":cys, "newfx":newfxs, "newfy":newfys, 
                    "newcx": newcxs, "newcy":newcys, "k1":k1s, "k2":k2s, "k3":k3s, "p1":p1s, 
                    "p2":p2s,}

    
    xs = parameterDict[parameter]

    mu, std = norm.fit(xs)
    # remove outliers
    newxs = [x for x in xs if ((x > mu - 3*std) and (x < mu + 3*std))]
    mu, std = norm.fit(newxs)

    plt.figure()

    # plot histogram of parameter
    plt.hist(newxs, bins=70, density=True, edgecolor="k")
    _, ymax = plt.ylim()
    plt.vlines(refParameter, 0, ymax, "r", linewidth=2, label="Reference Value = {:.2g}".format(refParameter))
    plt.vlines(mu, 0, ymax, "g", linewidth=2, label="Mean = {:.2g}".format(mu))

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 1000)
    y = norm.pdf(x, mu, std)

    plt.plot(x, y, color="m", ls="-", label="Gaussian Distribution")

    plt.xlabel(parameter+"s")
    plt.ylabel("Probability Density")

    plt.legend()

    plt.savefig("method_2_30000_graphs/{}_graph.png".format(parameter))

    return None

indexes = [275, 419, 492, 186, 622, 593, 529, 311, 285, 23, 129, 52, 379, 28, 193, 257, 301, 413, 581, 366, 602, 108, 230, 594, 483, 20, 8, 391, 584, 368, 380, 18, 167, 539, 446, 223, 502, 184, 515, 97, 162, 248, 355, 267, 22, 277, 151, 148, 172, 417, 47, 137, 219, 424, 133, 369, 24, 143, 261, 241, 244, 211, 300, 75, 385, 48, 509, 209, 99, 367, 128, 329, 527, 500, 94, 489, 64, 253, 378, 158, 19, 120, 119, 302, 232, 181, 605, 501, 436, 173, 316, 190, 388, 592, 123, 477, 389, 114, 472, 60, 595, 571, 63, 349, 441, 213, 608, 503, 67, 530, 351, 287, 523, 458, 202, 87, 269, 357, 603, 273, 387, 53, 332, 370, 227, 256, 418, 161, 263, 242, 614, 34, 433, 92, 610, 194, 507, 476, 442, 462, 68, 469, 583, 364, 45, 451, 513, 252, 416, 564, 175, 381, 309, 212, 0, 260, 350, 132, 578, 521, 562, 536, 473, 444, 390, 296, 558, 326, 214, 286, 518, 305, 243, 203, 168, 327, 89, 236, 38, 40, 557, 438, 96, 588, 428, 131, 330, 609, 611, 101, 488, 84, 113, 526, 125, 189, 589, 375, 467, 117]

# compareSample(pointFile, 10, 30000, 1920, 1080, indexes)

savefile = "parameters_method_2_(30000_repetitions).npz"

parameters = ["fx", "fy", "cx", "cy", "newfx", "newfy", 
                    "newcx", "newcy", "k1", "k2", "k3", "p1", 
                    "p2"]


# METHOD 1
for parameter in parameters:
    if parameter == "k2" or parameter == "k3":
        pass
    else:
        parameter += "s"
        subGroupAverages(loadFile, 500, parameter)

# subGroupAverages(loadFile, 2000, "k2s")


# METHOD 2
# for parameter in parameters:
#     Method2(savefile, parameter)



    
    











    





