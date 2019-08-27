import cv2

class Marker:

    def __init__(self, contour):

        # Contour
        self.contour = contour

        # Shape
        x, y, w, h = cv2.boundingRect(contour)
        self.x = x
        self.y = y
        self.w = w
        self.h = h
