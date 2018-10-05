#!/usr/bin/env python3
from PIL import Image
import numpy as np

def get_pix_val(x, y, image):
    im = Image.open(image, "r")     # open the image
    pix_vals = list(im.getdata())   # get the pixel values
    w, h = im.size                  # get width and height
    if im.mode == "RGB":
        channels = 3
    elif im.mode == "L":
        channels = 1
    else:
        print("Unknown mode: {}".format(im.mode))
        return None
    
    pix_vals = np.array(pix_vals).reshape((w,h,channels))     # reshape the array 

    return pix_vals[x][y]       # returns the RGB or Grey channel value of given pixel (x,y)


def main():
    im = Image.open("test.jpg", "r")
    w, h = im.size
    xar = np.zeros(w)
    yar = np.zeros(h)

    print(len(xar), len(yar))

    for x in range(w):
        for y in range(h):
            print("Pixel values for pixel ({},{})".format(x,y))
            # print(get_pix_val(x, y, "test.jpg"))
            print(im.getpixel((x,y)))






if __name__ == "__main__":
    main()
