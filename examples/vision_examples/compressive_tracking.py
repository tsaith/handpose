import sys
sys.path.append('..')

import cv2 # OpenCV
import time

from vision.core import Rect
from vision.tracking.ct import CompressiveTracker, detect_box
from vision.tracking.ct import PyCompressiveTracker
from vision.gui import wait_key
from vision.draw import draw_rect

import argparse


def parse_box_arg(arg):
    # Parse the box argument and return a Rect object

    out = arg
    out = out.replace('(', "")
    out = out.replace(')', "")
    out = out.replace(' ', "")
    out = out.split(',')
    
    x = int(out[0])
    y = int(out[1])
    width  = int(out[2])
    height = int(out[3])

    return Rect(x, y, width, height)


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default=None, help="path to the video file")
ap.add_argument("-b", "--box", default=None, help="detecting box, e.g. (10, 10, 20, 20)")
args = vars(ap.parse_args())

# Arguments
video = args['video']
box = parse_box_arg(args['box']) if args['box'] else None 

# Read frames form webcam or video file
if video is None:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480);

else:
    cap = cv2.VideoCapture(video)

# In case the video can't be open
if not cap.isOpened(): 
    print("The video cann't be open!")
    exit(0)

# Creat a window
win_name = 'demo'
cv2.namedWindow(win_name)


if video is None: # When webcan is used
    # Define the detection box
    box, first_frame = detect_box(cap, win_name)
else:
    _, first_frame = cap.read()

# Initialize the compresive tracker
gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
#ct = PyCompressiveTracker(gray, box)
ct = CompressiveTracker(gray, box)

# loop over the frames of the video
print("Start to track the object.")
while True:
    # grab the current frame and initialize the occupied/unoccupied
    # text
    (grabbed, frame) = cap.read()

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not grabbed: break

    # Process the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    box = ct.process_frame(gray, box)

    # Draw the tracking box
    draw_rect(frame, box, color=(0, 255, 0), thickness=2)

    # show the frame and record if the user presses a key
    cv2.imshow(win_name, frame)

    # if the `q` key is pressed, break from the lop
    key = wait_key(1)
    if key == ord("q"): break


# Release the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
