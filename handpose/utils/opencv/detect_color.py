import numpy as np
import cv2

def detect_color(image, color='blue'):
    # detect color from a image.

    # converting from BGR to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if color == 'blue':

        # Range for blue color
        lower_color = np.array([110,50,50])
        upper_color = np.array([130,255,255])

    elif color == 'green':

        # Range for green color
        lower_color = np.array([45, 50, 50])
        upper_color = np.array([90, 255, 255])

    else:
        print('Error: invalid color {} is set.'.format(color))

    mask = cv2.inRange(hsv, lower_color, upper_color)
    out = cv2.bitwise_and(image, image, mask=mask)

    return out, mask
