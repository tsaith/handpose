import numpy as np
import matplotlib.pyplot as plt
from .color import gray2rgb
from .core import Point, Contour
import sys

def contour_points(f):
    # Return contour points
    # where f is the input image with binary levels.
    # Assuming z is the contour array,
    # set z[i, j] as one if any ajacent element around f[i, j] has non-zero value

    rows, cols = f.shape
    z = np.zeros(shape = (rows, cols), dtype = np.bool)
    for i in np.arange(1, rows-1):
        for j in np.arange(1, cols-1):
            if f[i, j] == 0:
                if (f[i-1, j] == 1 or
                    f[i+1, j] == 1 or
                    f[i, j-1] == 1 or
                    f[i, j+1] == 1 or
                    f[i-1, j-1] == 1 or
                    f[i+1, j-1] == 1 or
                    f[i-1, j+1] == 1 or
                    f[i+1, j+1] == 1):

                    z[i, j] = 1

    return z

def contour_indices(z):
    # Return the contour indices
    # where z is the contour array

    rows, cols = z.shape
    ci = np.zeros(shape = (rows, cols), dtype = np.int32)
    ci.fill(-1) # Set initail value as -1
    current_ci = -1

    # Set initial value for contour indexess
    for i in np.arange(1, rows - 1):
        for j in np.arange(1, cols - 1):
            if z[i, j] == True: # coutour point
                current_ci = set_init_ci(i, j, ci, current_ci)

    # Merge contour indexes
    for i in np.arange(1, rows - 1)[::-1]: # down-up
        for j in np.arange(1, cols - 1):
            diffuse_ci(i, j, ci)

    for i in np.arange(1, rows - 1):       # up-down
        for j in np.arange(1, cols - 1):
            diffuse_ci(i, j, ci)

    for i in np.arange(1, rows - 1)[::-1]: # down-up again
        for j in np.arange(1, cols - 1):
            diffuse_ci(i, j, ci)
    # Reorder contour indexes
    reorder_ci(ci)

    return ci


def set_init_ci(i, j, ci, current_ci):
    # Set initial value of contour indexes and
    # return current contour index

    ajacent_ci = np.zeros(shape = (4), dtype = np.int32)

    ishift = np.array([0, -1 , -1, -1])
    jshift = np.array([-1, -1, 0, 1])

    # get ajacent contour indexess
    for k in np.arange(4):
        ii = i + ishift[k]
        jj = j + jshift[k]
        ajacent_ci[k] = ci[ii, jj]

    ajacent_ci_logic_array = ajacent_ci > -1
    ci_candidates = ajacent_ci[ajacent_ci_logic_array]
    have_ajacent_ci = True if ci_candidates.size > 0 else False
    if have_ajacent_ci:
        ci_min = ci_candidates.min()
        for k in np.arange(4):
            if ajacent_ci_logic_array[k]:
               ii = i + ishift[k]
               jj = j + jshift[k]
               ci[ii, jj] = ci_min
        ci[i, j] = ci_min
    else:
        current_ci += 1
        ci[i, j] = current_ci

    return current_ci


def diffuse_ci(i, j, ci):
    # Diffuse the contour indices array

    ci_window = ci[i-1: i+2, j-1: j+2]
    ci_logic_array = ci_window > -1
    ci_candidates = ci_window[ci_logic_array]
    have_valid_ci = True if ci_candidates.size > 0 else False
    if have_valid_ci:
        ci_min = ci_candidates.min()
        for ii in np.arange(3):
            for jj in np.arange(3):
                if ci_logic_array[ii, jj]:
                    ci[i+ii-1, j+jj-1] = ci_min


def reorder_ci(array):
    # Reorder the contour indexes

    rows, cols = array.shape

    ordered_list = unique_list(array).tolist()
    for i in np.arange(rows):
        for j in np.arange(cols):
            index = ordered_list.index(array[i, j])
            array[i, j] = index - 1


def unique_list(array, min_val = None, max_val = None):
    # Return the list containing unique elements

    if min_val == None:
        min_val = array.min()

    if max_val == None:
        max_val = array.max()

    out = np.unique(array)
    logic_array = (out >= min_val) * (out <= max_val)
    out = out[logic_array]

    return out

def get_contours(image):
    # Return the list of contour objects
    # where the image is binary

    rows, cols = image.shape

    z = contour_points(image)
    ci = contour_indices(z)

    contour_number = ci.max() + 1
    contours = []
    for index in np.arange(contour_number):
        contours.append(Contour(index))

    # classify contour points
    for i in np.arange(1, rows - 1):
        for j in np.arange(1, cols - 1):
            index = ci[i, j]
            if index > -1:
                x = j
                y = i
                point = Point(x, y)
                contour = contours[index]
                contour.append(point)

    # Find ends of contours
    for index in np.arange(len(contours)):
        contours[index].set_ends()

    return contours

def filter_contours(contours_in, np_min = None, np_max = None, width_min = None, width_max = None, height_min = None, height_max = None):
    # Return the contours filtered by some criterions
    # np_min: minimun number of points
    # np_max: maximun number of points

    contours = np.copy(contours_in)

    num = len(contours)

    if np_min != None:
        contours = [contour for contour in contours if contour.point_num() >= np_min ]

    if np_max != None:
        contours = [contour for contour in contours if contour.point_num() <= np_max ]

    if width_min != None:
        contours = [contour for contour in contours if contour.width >= width_min ]

    if width_max != None:
        contours = [contour for contour in contours if contour.width <= width_max ]

    if height_min != None:
        contours = [contour for contour in contours if contour.height >= height_min ]

    if height_max != None:
        contours = [contour for contour in contours if contour.height <= height_max ]

    return contours


def draw_contours(contours, image, color = 'red'):
    # Draw contours

    ndim = image.ndim
    if ndim == 3: # color
        image_rgb = np.copy(image)
    elif ndim == 2: # gray-scale
        image_rgb = gray2rgb(image)
    else:
        print("Error: ndim = ", ndim)

    if color == 'red':
        pixel = np.array([255, 0, 0]) # Red
    elif color == 'green':
        pixel = np.array([0, 255, 0]) # Green
    else:
        pixel = np.array([0, 0, 255]) # Blue

    # Set color
    for contour in contours:
        for point in contour.points:
            x = int(point.x + 0.5)
            y = int(point.y + 0.5)
            image_rgb[y, x, :] = pixel

    return image_rgb

def draw_points(points, image, color = 'green'):
    # Draw points

    ndim = image.ndim
    if ndim == 3: # color
        image_rgb = np.copy(image)
    elif ndim == 2: # gray-scale
        image_rgb = gray2rgb(image)
    else:
        print("Error: ndim = ", ndim)

    if color == 'red':
        pixel = np.array([255, 0, 0]) # Red
    elif color == 'green':
        pixel = np.array([0, 255, 0]) # Green
    else:
        pixel = np.array([0, 0, 255]) # Blue

    # Set color
    for point in points:
        x = int(round(point.x))
        y = int(round(point.y))
        image_rgb[y, x, :] = pixel

    return image_rgb

def get_line_points(p1, p2):
    """
    Return the points of one line.
    If the line is more horizontal,
    it is expressed as y = a*x + b, else, as x = a'*y + b'.
    """

    x1 = p1.x
    y1 = p1.y
    x2 = p2.x
    y2 = p2.y

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    points = []

    if dx > dy: # line is more horizontal

        a = (y2 - y1) / (x2 - x1)
        b = y1 - a*x1

        ix1 = int(x1 + 0.5)
        ix2 = int(x2 + 0.5)

        if ix2 > ix1:
            ixa = ix1
            ixz = ix2
        else:
            ixa = ix2
            ixz = ix1

        for ix in np.arange(ixa, ixz+1):
            x = 1.0*ix
            y = a*x + b
            points.append(Point(x, y))
    else:
        a = (x2 - x1) / (y2 - y1)
        b = x1 - a*y1

        iy1 = int(y1 + 0.5)
        iy2 = int(y2 + 0.5)

        if iy2 > iy1:
            iya = iy1
            iyz = iy2
        else:
            iya = iy2
            iyz = iy1

        for iy in np.arange(iya, iyz+1):
            y = 1.0*iy
            x = a*y + b
            points.append(Point(x, y))

    return points

def locate_contour(indexes, ci, image_g):

    rows, cols = image_g.shape
    channels = 3
    image_rgb = np.zeros(shape = (rows, cols, channels), dtype = np.uint8)

    red_pxel = np.array([255, 0, 0])
    green_pxel = np.array([0, 255, 0])
    for i in np.arange(1, rows-1):
        for j in np.arange(1, cols-1):
            image_rgb[i, j, :] = image_g[i, j]

    # locate the contour corresponding to the specific contour indexes
    for i in np.arange(1, rows-1):
        for j in np.arange(1, cols-1):
            if ci[i, j] > 1:
                if ci[i, j] == indexes:
                    image_rgb[i, j, :] = red_pxel
                else:
                    image_rgb[i, j, :] = green_pxel

    return image_rgb

def find_contours_by_indexes(indexes, contours_in):
    # Find contours by indexes

    contours = []
    for index in indexes:
        contours.append(contours_in[index])

    return contours

def check_bound1(x, xa, xz):
    """
    Check if the x lies in (xa, xz), 1D.
    x: x value.
    xa, xz: the boundries.
    """
    out = False
    if x > xa and x < xz: out = True

    return out
