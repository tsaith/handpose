import numpy as np
import sys
from .lib.libcore import Point, Rect

class Point:

    def __init__(self, x, y):
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, v):
        self._x = v

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, v):
        self._y = v

"""

class Rect:

    def __init__(self, x, y, width, height):

        self.x = x
        self.y = y
        self.width = width
        self.height = height

"""

class Rectangle:

    def __init__(self, pt1, pt2):
        self._pt1 = pt1
        self._pt2 = pt2

    @property
    def pt1(self):
        return self._pt1

    @property
    def pt2(self):
        return self._pt2

    @property
    def width(self):
        x1 = self.pt1.x
        x2 = self.pt2.x
        width = x2 - x1 + 1

        return width

    @property
    def height(self):
        y1 = self.pt1.y
        y2 = self.pt2.y
        height = y2 - y1 + 1

        return height

    @property
    def area(self):
        area = self.width * self.height
        return area

    @property
    def shape(self):
        x1 = self.pt1.x
        y1 = self.pt1.y
        x2 = self.pt2.x
        y2 = self.pt2.y

        rows = int(y2 - y1 + 1.5)
        cols = int(x2 - x1 + 1.5)
        shape = (rows, cols)

        return shape

class BBox(Rectangle):
    """
    Bonding box.
    """
    pass

class Contour:

    def __init__(self, index):
        self.index = index
        self.points = []
        self._center_point = None
        self.x_min = 0.0
        self.x_max = 0.0
        self.y_min = 0.0
        self.y_max = 0.0
        self._width = None
        self._height = None

    def append(self, point):
        self.points.append(point)

    def pop(self):
        self.points.pop()

    def point_num(self):
        return len(self.points)

    def set_ends(self):
        """
        Set ends
        """

        n = self.point_num()

        self.x_min = sys.maxsize
        self.y_min = sys.maxsize

        for index in np.arange(n):
            point = self.points[index]
            x = point.x
            y = point.y

            if self.x_min > x:
                self.x_min = x
            if self.x_max < x:
                self.x_max = x

            if self.y_min > y:
                self.y_min = y
            if self.y_max < y:
                self.y_max = y

    def set_center_point(self):
        """
        Set the center point.
        """
        x = 0.5*(self.x_min + self.x_max)
        y = 0.5*(self.y_min + self.y_max)
        self._center_point = Point(x, y)

    def set_attributes(self):
        """
        Set attributes.
        """

        self.set_ends()
        self.set_center_point()

    def merge(self, contour):
        """
        Merge the other contour.
        """
        self.points += contour.points
        self.set_attributes()

        return self

    @property
    def width(self):
        if self._width == None:
            self._width = self.x_max - self.x_min + 1
        return self._width

    @property
    def height(self):
        if self._height == None:
            self._height = self.y_max - self.y_min + 1
        return self._height

    @property
    def center_point(self):

        if self._center_point == None:
            self.set_center_point()

        return self._center_point

    @property
    def left_point(self):
        # Return the left point in x direction

        x = self.x_min
        y = self.center_point.y

        return Point(x, y)

    @property
    def right_point(self):
        # Return the right point in x direction

        x = self.x_max
        y = self.center_point.y

        return Point(x, y)

    @property
    def upper_point(self):
        # Return the upper point in y direction

        x = self.center_point.x
        y = self.y_min

        return Point(x, y)

    @property
    def lower_point(self):
        # Return the lower point in y direction

        x = self.center_point.x
        y = self.y_max

        return Point(x, y)

class SlidingWindow:

    def __init__(self, x0, y0, width, height, image):
        self._x0 = x0
        self._y0 = y0
        self._width = width
        self._height = height
        self._image = image
        self._shape = None

    @property
    def x0(self):
        return self._x0

    @property
    def y0(self):
        return self._y0

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def image(self):
        return self._image

    @property
    def shape(self):
        self._shape = self._image.shape
        return self._shape


def list2str(list_in):
    """
    Convert a list to a string.
    """
    return ''.join(list_in)

def rand_uniform(low, high, size=None):
    """
    Generate a random number for uniform distribution,
    which lies in [low, high).
    """

    r = np.random.random_sample(size=size)
    out = low + r * (high-low)

    return out

def floor(val):
    out = np.int(np.floor(val))
    return out

def ceil(val):
    out = np.int(np.ceil(val))
    return out

def round(val):
    out = np.int(np.round(val))
    return out
