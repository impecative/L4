from __future__ import division         # compatibility for Python 2.x 
import numpy as np
import time, sys

### Given a set {x_i} of homogeneous coordinates (x,y,w)^T, compute
### the translation and scale transformation such that the following
### criteria are met:
###     1) the centroid of the set of points is at the origin.
###     2) the points are scales such that their average distance 
###        from the origin is sqrt(2). i.e their average point is 
###        (1,1,1)^T

# example points, input style...
points = np.array(((1,1), (1,2), (3,1), (4,-1), (6, 2), (5,2), (1,0)))

def centroid(arr):
    """For a set of coordinates, compute the centroid position."""
    length = arr.shape[0]      # length of the array
    sum_x  = np.sum(arr[:,0])
    sum_y  = np.sum(arr[:,1])

    return sum_x/length, sum_y/length

def translation(arr):
    """Produces a new set of points that has undergone a translation such that 
    the centroid of the original set of points now lies at the origin."""
    # OLD SLOW...
    # x_c, y_c = centroid(arr)
    # newpoints = np.zeros(arr.shape)
    # for i in range(arr.shape[0]):
    #     newpoints[i][0] = arr[:,0][i] - x_c
    #     newpoints[i][1] = arr[:,1][i] - y_c
    # return newpoints

    ## NEW FAST
    x_c, y_c = centroid(arr)
    newpoints = np.array((arr[:,0]-x_c, arr[:,1]-y_c))
    return newpoints.T
   

def avgDisplacement(arr):
    """Finds the average displacement of a set of points (x_i, y_i)
    from the origin."""
    sum_squares = np.sum(arr**2, axis=1)
    return np.mean(np.sqrt(sum_squares))    

def scale(arr):
    """Fixes the set of points (x_i, y_i) such that the average 
    displacement from the origin is sqrt(2)."""
    displacement = avgDisplacement(arr)
    scale = np.sqrt(2)/displacement

    newpoints = scale * arr

    #assert np.isclose(avgDisplacement(newpoints), np.sqrt(2)), "Scaling has not worked correctly..."

    return newpoints

def normalisation(arr):
    """Given a set of points {x_i}, normalise them such that\n 
    a) the centroid of the group is at the origin \n
    b) the average displacement of the group is sqrt(2)\n
    return: The new set of normalised points corresponding to 
    original set of points."""

    # 1) translate the points such that the centroid is at origin
    points = translation(arr)
    # 2) scale the points such that average displacement is sqrt(2)
    newpoints = scale(points)

    return newpoints

# more example data
data = np.random.randint(0, 10, size=(10000, 2))

newpoints = normalisation(data)
print(centroid(newpoints))
print(avgDisplacement(newpoints))
    
    