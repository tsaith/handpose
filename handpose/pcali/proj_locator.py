from .point import Point
from .marker import Marker
from ..geometry import *
from ..utils.opencv import detect_color

import numpy as np
import cv2
#import imutils

def grab_contours(cnts):
    # if the length the contours tuple returned by cv2.findContours
    # is '2' then we are using either OpenCV v2.4, v4-beta, or
    # v4-official
    if len(cnts) == 2:
        cnts = cnts[0]

    # if the length of the contours tuple is '3' then we are using
    # either OpenCV v3, v4-pre, or v4-alpha
    elif len(cnts) == 3:
        cnts = cnts[1]

    # otherwise OpenCV has changed their cv2.findContours return
    # signature yet again and I have no idea WTH is going on
    else:
        raise Exception(("Contours tuple must have length 2 or 3, "
            "otherwise OpenCV changed their cv2.findContours return "
            "signature yet again. Refer to OpenCV's documentation "
            "in that case"))

    # return the actual contours array
    return cnts

class ProjLocator:
    # Projective locator.

    def __init__(self, image):

        self.image = image
        self.image_h, self.image_w, self.image_c = image.shape
        self.image_cx = 0.5*self.image_w
        self.image_cy = 0.5*self.image_h

        self.tp_markers = None
        self.sc_markers = None

        self.tp_points_px = None # In pixel coordicates
        self.sc_points_px = None
        self.tp_points = self.get_tp_points() # In Earth coordicates (Normalied)
        self.sc_points = None


    def get_binary_image(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        #binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,5)
        ret, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)


        return binary


    def find_markers(self, image):

        # Image size
        h, w, c = image.shape
        image_area = h*w

        # Binary image
        binary = self.get_binary_image(image)

        # All contours
        cnts = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = grab_contours(cnts)
        #cnts = imutils.grab_contours(cnts)

        # Get marker and contour
        markers = []
        for c in cnts:

            area = cv2.contourArea(c)
            if area < image_area*0.0001 or area > image_area*0.01:
                continue

            x, y, w, h = cv2.boundingRect(c)
            ratio = 1.0*w/h

            if ratio > 2.0 or ratio < 0.5:
                continue

            marker = Marker(c)
            markers.append(marker)

        return markers

    def detect_tp_markers(self, image):

        detected, mask = detect_color(image.copy(), color='blue')

        markers = self.find_markers(detected)

        return markers


    def detect_sc_markers(self, image):

        detected, mask = detect_color(image.copy(), color='green')

        markers = self.find_markers(detected)

        return markers


    def detect_markers(self):

        image = self.image

        tp_markers = self.detect_tp_markers(image)
        sc_markers = self.detect_sc_markers(image)

        # Sort markers
        tp_markers = self.sort_markers(tp_markers)
        sc_markers = self.sort_markers(sc_markers)

        # Save
        self.tp_markers = tp_markers
        self.sc_markers = sc_markers

        return tp_markers, sc_markers


    def sort_markers(self, markers_in):
        # Sort markerscounter-clockwise

        cx, cy = self.image_cx, self.image_cy

        groups = [[] for i in range(4)] # Counter-clockwise
        for marker in markers_in:

            if marker.x < cx and marker.y < cy:
                groups[0].append(marker)

            if marker.x < cx and marker.y > cy:
                groups[1].append(marker)

            if marker.x > cx and marker.y > cy:
                groups[2].append(marker)

            if marker.x > cx and marker.y < cy:
                groups[3].append(marker)


        # Markers sorted
        markers_out = []
        for group in groups:
            markers_out.append(group[0])

        return markers_out


    def draw_image(self, line_width=10):

        image = self.image.copy()
        h, w, c = image.shape

        # TP
        for marker in self.tp_markers:
            p1 = (marker.x, marker.y)
            p2 = (marker.x+marker.w, marker.y+marker.h)
            cv2.rectangle(image, p1, p2, (255, 0, 0), line_width)

        # Screen
        for marker in self.sc_markers:
            p1 = (marker.x, marker.y)
            p2 = (marker.x+marker.w, marker.y+marker.h)
            cv2.rectangle(image, p1, p2, (0, 255, 0), line_width)

        # Reference point of TP
        for p in self.tp_points_px:
            x = int(p.x+0.5)
            y = int(p.y+0.5)
            cv2.circle(image, (x, y), 15, (0, 0, 255), -1)

        # Reference point of screen
        for p in self.sc_points_px:
            x = int(p.x+0.5)
            y = int(p.y+0.5)
            cv2.circle(image, (x, y), 15, (0, 0, 255), -1)

        return image

    def get_tp_points(self):

        xa = 0.0
        xz = 1.0
        ya = 0.0
        yz = 1.0

        p0 = Point(xa, ya)
        p1 = Point(xa, yz)
        p2 = Point(xz, yz)
        p3 = Point(xz, ya)

        points = [p0, p1, p2, p3]

        return points

    def get_tp_points_px(self, markers):

        x = markers[0].x + markers[0].w
        y = markers[0].y + markers[0].h
        p0 = Point(x, y)

        x = markers[1].x + markers[1].w
        y = markers[1].y
        p1 = Point(x, y)

        x = markers[2].x
        y = markers[2].y
        p2 = Point(x, y)

        x = markers[3].x
        y = markers[3].y + markers[3].h
        p3 = Point(x, y)

        points = [p0, p1, p2, p3]

        return points

    def get_sc_points_px(self, markers):

        points = []
        for i, marker in enumerate(markers):

            if i == 0: # For p1
                x = marker.x
                y = marker.y

            if i == 1: # For p2
                x = marker.x
                y = marker.y + marker.h

            if i == 2: # For p3
                x = marker.x + marker.w
                y = marker.y + marker.h

            if i == 3: # For p4
                x = marker.x + marker.w
                y = marker.y

            points.append(Point(x,y))

        return points

    def get_sc_point_component(self, p0, p1, p2):

        x0 = p0.x
        y0 = p0.y
        x1 = p1.x
        y1 = p1.y
        x2 = p2.x
        y2 = p2.y

        d = get_distance_from_two_points(x1, y1, x2, y2)
        xp, yp = get_perp_foot_from_three_points(x0, y0, x1, y1, x2, y2)
        d1 = get_distance_from_two_points(x1, y1, xp, yp)

        out = d1/d * 1.0

        return out


    def get_sc_points(self):

        # Find the tp points in pixel space
        tp_points_px = self.get_tp_points_px(self.tp_markers)

         # Find the sc points in pixel space
        sc_points_px = self.get_sc_points_px(self.sc_markers)
        sc_points = []

        tp_p0_px = tp_points_px[0]
        tp_p1_px = tp_points_px[1]
        tp_p2_px = tp_points_px[2]
        tp_p3_px = tp_points_px[3]

        sc_p0_px = sc_points_px[0]
        sc_p1_px = sc_points_px[1]
        sc_p2_px = sc_points_px[2]
        sc_p3_px = sc_points_px[3]

        # For p0
        x = self.get_sc_point_component(sc_p0_px, tp_p0_px, tp_p3_px)
        y = self.get_sc_point_component(sc_p0_px, tp_p0_px, tp_p1_px)
        sc_points.append(Point(x, y))

        # For p1
        x = self.get_sc_point_component(sc_p1_px, tp_p1_px, tp_p2_px)
        y = self.get_sc_point_component(sc_p1_px, tp_p0_px, tp_p1_px)
        sc_points.append(Point(x, y))

        # For p2
        x = self.get_sc_point_component(sc_p2_px, tp_p1_px, tp_p2_px)
        y = self.get_sc_point_component(sc_p2_px, tp_p3_px, tp_p2_px)
        sc_points.append(Point(x, y))

        # For p3
        x = self.get_sc_point_component(sc_p3_px, tp_p0_px, tp_p3_px)
        y = self.get_sc_point_component(sc_p3_px, tp_p3_px, tp_p2_px)
        sc_points.append(Point(x, y))

        # Save points
        self.tp_points_px = tp_points_px
        self.sc_points_px = sc_points_px
        self.sc_points = sc_points

        return sc_points
