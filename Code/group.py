import sys
import os


def series(k, m):
    if not m%2 == 0:            # check m is odd 

        # check not going into negative numbers! 
        offset = int((m-1)/2)

        if k - offset < 0:  # this may need to be <=
            print("{} is too small for centred series of {}".format(k, m))
            return None

        originalOffset = offset
        group = []

        while abs(offset) <= originalOffset: 
            group.append(k-offset)
            offset -= 1
        
        return group
    else:
        print("m must be odd!")
        return None


print(series(3,7))

path = "C:/Users/alexe/Pictures"  # example path

flist = os.listdir(path)

print(flist)