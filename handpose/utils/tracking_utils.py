import numpy as np
import cv2 # OpenCV


def get_bbox_center(frame_width, frame_height, bbox_width, bbox_height):
    """
    Return the bbox at center.
    """
    xc = 0.5 + 0.5*frame_width
    yc = 0.5 + 0.5*frame_height

    x = round(xc - 0.5*bbox_width)
    y = round(yc - 0.5*bbox_height)

    bbox = (x, y, bbox_width, bbox_height)

    return bbox

