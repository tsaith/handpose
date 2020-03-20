import sys
import os
import numpy as np
from touch.utils.file import to_str_digits
from touch.utils.image import make_square
from touch.utils.opencv import Webcam, wait_key
import cv2 as cv
import face_recognition


# Load the cascade
## For windows
face_cascade = cv.CascadeClassifier('C:\Python37\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml')
body_cascade = cv.CascadeClassifier('C:\Python37\Lib\site-packages\cv2\data\haarcascade_upperbody.xml')
## For Ubuntu
#face_cascade = cv.CascadeClassifier('/home/andrew/.local/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_alt.xml')


def detect_faces(frame):

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # Detect the faces
    face_locations = face_recognition.face_locations(frame_rgb)
    faces = to_locations(face_locations)

    return faces


def detect_faces_opencv(frame):

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    return faces

def detect_bodies_opencv(frame):

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    bodies = body_cascade.detectMultiScale(gray, 1.1, 4)

    return bodies


def get_image_index(num_images, ratio):

    index = int(ratio*(num_images-1))

    return index


def to_locations(face_locations):
    # Convert to the format of [(left, top, width, height)].

    locations = []

    for (top, right, down, left) in face_locations:

        width = right - left
        height = down - top

        loc = (left, top, width, height)
        locations.append(loc)

    return locations

def draw_faces(image, locations, color=(255, 0, 0)):

    drawn = image.copy()
    for (left, top, width, height) in locations:
        p_start = (left, top)
        p_end = (left+width, top+height)
        drawn = cv.rectangle(drawn, p_start, p_end, color, 2)

    return drawn


def get_toroidal_angle(x, width):

    angle_range = 1.0/3.0*np.pi # 120 degree
    ratio = 1.0*x / width

    angle = (ratio - 0.5) * angle_range

    return angle

def to_degree(angle):
    return angle*180.0/np.pi


def get_target_id(locations):

    areas = []
    for (left, top, width, height) in locations:

        area = width*height
        areas.append(area)

    areas = np.array(areas)

    if len(areas) > 0:
        target_id = np.argmax(areas)
    else:
        target_id = None

    return target_id
