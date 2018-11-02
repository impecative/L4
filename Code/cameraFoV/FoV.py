import numpy as np 
import matplotlib.pyplot as plt 
import random

def lineA(x, gamma):
    ## Note: gamma is in radians!!
    return -np.tan(gamma)*x

def lineB(x, delta, b):
    ## Note: delta is in radians!!
    return np.tan(delta)*(x + b)

def lineC2(x, gamma):
    return np.tan(gamma)*x

def lineD(x, delta, b):
    return -np.tan(delta)*(x + b)


def test(x, z, baseline, FoV1, FoV2):
    """Given an coordinate in the x,z-plane, can two cameras
    of two different fields of view (FoV1, FoV2) in degrees 
    both see coordinate (x,z)?"""

    ret = False

    alpha = FoV1/2.         # in degrees
    beta = FoV2/2.          # in degrees

    gamma = np.pi/2 - np.deg2rad(alpha)     # in radians 
    delta = np.pi/2 - np.deg2rad(beta)      # in radians

    return x, z, baseline, gamma, delta


def plot(xs, zs, baseline, gamma, delta, z=""):
    if gamma > delta:
        print("We need wider angle camera to be in position (-b,0)!")
        print("Check the cameras are the correct way around...")
        return None

    xRange = 0
    if z:
        z = int(z)
        if z >= (baseline*np.tan(delta)*np.tan(gamma))/(np.tan(delta)-np.tan(gamma)):
            xRange = (z/np.tan(delta) - baseline) - ((-z)/np.tan(delta) - baseline)

        elif z >= (baseline*np.tan(delta)*np.tan(gamma))/(np.tan(delta) + np.tan(gamma)):
            xRange = (z/np.tan(delta) - baseline) - (-z)/np.tan(gamma)
        else:
            pass

    if not xRange == 0:
        print("The maximum width of birds at {:.1f} m is {:.1f} m".format(z, xRange))
    if not z:
        None
    elif z and xRange == 0:
        print("No birds can be observed at distance of {:.1f} m".format(z))
            
    ALine = []
    BLine = []
    CLine = []
    DLine = []

    for X in np.arange(-100, 0.5, 50):
        ALine.append(lineA(X, gamma))

    for X in np.arange(-baseline, 100, 50):
        BLine.append(lineB(X, delta, baseline))

    for X in np.arange(-baseline-50, -baseline+2,50):
        DLine.append(lineD(X, delta, baseline))

    for i in np.arange(0,51,50):
        CLine.append(lineC2(i, gamma))

    # plot the cameras and stuff
    plt.figure()

    plt.plot(np.arange(-100, 0.5, 50), ALine, "g--", alpha=0.8)
    plt.plot(np.arange(-baseline, 100, 50), BLine, "b--", alpha=0.8)
    plt.plot(np.arange(-baseline-50, -baseline+2, 50), DLine, "b--", alpha=0.8)
    plt.plot(np.arange(0, 51, 50), CLine, "g--", alpha=0.8)

    # draw vertical lines at camera centres:
    plt.vlines(0, 0, 100, colors="k", alpha=0.1)
    plt.vlines(-baseline, 0, 100, colors="k", alpha=0.1)
    plt.hlines(0, -50, 50, colors="k", alpha = 0.1)

    # draw the two cameras
    plt.scatter((-baseline, 0), (0, 0), color="k", marker="s", s=40)

    # label the cameras
    plt.annotate("Camera 1", xy=(0,0), xytext=(0-3,-6),  
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

    plt.annotate("Camera 2", xy=(-baseline,0), xytext=(-baseline-3,-6),  
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))




    for ix, x in enumerate(xs):
        for iz, z in enumerate(zs):
            if (z >= (baseline*np.tan(delta)*np.tan(gamma))/(np.tan(delta)+np.tan(gamma))) and (z <= 
            (baseline*np.tan(delta)*np.tan(gamma)/(np.tan(delta)-np.tan(gamma)))):
                if (x >= -z/np.tan(gamma)) and (x <= (z/np.tan(delta) - baseline)):
                    ret = True
                else:
                    ret = False

            elif z > (baseline*np.tan(delta)*np.tan(gamma))/(np.tan(delta)-np.tan(gamma)):
                if (x >= (-z/np.tan(delta) - baseline)) and (x <= (z/np.tan(delta)) - baseline):
                    ret = True
                else:
                    ret = False

            else:
                ret = False

            #print(ret)

            if ret:
                #plt.scatter(x, z, color="r", marker="x")
                None
            else:
                #plt.scatter(x, z, color="k", marker="x")
                None
    plt.xlim(-baseline-20, 10)
    plt.ylim(-10, 50)

    plt.show()

    return None


xs = []
zs = []
for i in range(25):
    xs.append(random.uniform(-30, 20))
    zs.append(random.uniform(0, 50))

# 20 mm lens has FoV 60.7 degrees
# 35 mm lens has FoV 37 degrees
# 50 mm lens has FoV 26.3

x, z, baseline, gamma, delta = test(12, 48, 25, 26.3, 26.3)    

plot(xs[:2], zs[:2], baseline, gamma, delta, z=150)

    


