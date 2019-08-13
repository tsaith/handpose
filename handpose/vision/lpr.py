import numpy as np
import sys

from .core import Point, Contour, list2str
from .color import rgb2gray, gray2binary, binary2gray
from .io import *
from .measure import (get_contours, check_bound1,
    find_contours_by_indexes)
from .ml.linear_model import logistic_predict_proba

import ipdb


class PredictModel:
    """
    Predict model which is used to predict class label.
    """

    def __init__(self, labels, coef):
        """
        labels: array-like, shape (n_classes)
        coef: array-like, shape (n_classes, n_features)
        """

        self._labels = labels
        self._coef = coef

    def predict(self, x):
        """
        Predict class label.
        """

        n_classes = self._coef.shape[0]

        proba = self.predict_proba(x)
        class_index = np.argmax(proba)
        label = self._labels[class_index]

        return label

    def predict_proba(self, x):
        """
        Probability estimates.
        """

        n_classes = self._coef.shape[0]

        proba = np.zeros(n_classes)
        for i in np.arange(n_classes):
            proba[i] = logistic_predict_proba(x, self._coef[i, :])

        return proba

def find_aligned_targets(contours):
    """
    Find out the aligned targets
    """

    height_min = 10 # Minimum value of target's height

    num = len(contours)
    indexes = []
    for i in np.arange(num):
        c1 = contours[i]
        dx_max = c1.height * 10.0
        dy_max = c1.height * 0.5

        c1_xc = c1.center_point.x
        c1_yc = c1.center_point.y

        if c1.height < height_min:
            continue

        aspect_ratio = c1.width / c1.height
        if aspect_ratio < 0.1 or aspect_ratio > 1.0:
            continue

        for j in np.arange(num):
            if j == i: continue
            c2 = contours[j]
            c2_xc = c2.center_point.x
            c2_yc = c2.center_point.y
            dxc = c2_xc - c1_xc
            dyc = c2_yc - c1_yc

            if dxc < 0 or dxc > c1.width*2.0: continue
            if abs(dyc) > dy_max: continue

            if c2.height < 0.5 * c1.height:
                continue

            if c2.height > 2.0 * c1.height:
                continue

            c2_ratio = c2.width / c2.height
            if c2_ratio < 0.1 or aspect_ratio > 1.0:
                continue

            tmp_indexes = []
            tmp_indexes.append(c1.index)
            tmp_indexes.append(c2.index)

            slope = (c2_yc - c1_yc) / (c2_xc - c1_xc)

            for k in np.arange(num):
                if k == i or k == j: continue

                ck = contours[k]
                ck_xc = ck.center_point.x
                ck_yc = ck.center_point.y
                dxc = ck_xc - c1_xc
                dyc = ck_yc - c1_yc

                if dxc < 0 or dxc > dx_max: continue
                if abs(dyc) > dy_max: continue

                if ck.height < 0.5 * c1.height:
                    continue

                if ck.height > 2.0 * c1.height:
                    continue


                ck_ratio = ck.width / ck.height
                if ck_ratio < 0.1 or aspect_ratio > 1.0:
                    continue

                y = find_y_by_point_and_slope(ck_xc, c1.center_point, slope)
                if abs(y - ck_yc) > dy_max:
                    continue

                tmp_indexes.append(ck.index)

            # Save indexes
            target_num = len(tmp_indexes)
            if len(indexes) < target_num:
                indexes = tmp_indexes

    return indexes

def find_y_by_point_and_slope(x, p0, slope):
    """
    Find y along one line,
    where
    x: x coordinate.
    p0: reference point.
    slope: slope of line.
    """

    x0 = p0.x
    y0 = p0.y

    y = y0 + slope * (x - x0)

    return y

def find_point_along_line(p0, s, d, sign = 0):
    """
    Find a point along line,
    where
    p0: reference point.
    s: slope of line.
    d: distance bween p0 and the new point.
    sign: 1 or -1
    """
    x0 = p0.x
    y0 = p0.y

    if sign == 0:
        print("Error: the value of sign should be 1 or -1")
        sys.exit()

    fac = d / np.sqrt(1.0 + s*s)
    x = x0 + sign * fac
    y = y0 + sign * s*fac

    return Point(x, y)

def linear_interp1(xa, xb, va, vb, xq):
    """
    1-D linear interpolation.
    xa, xb: lower and high x coordinate.
    va, vb: values corresponding to xa and xb
    xq: queried x.
    """
    r = (xb - xq) / (xb - xa)
    vq = r * va + (1.0 - r) * vb

    return vq

def sort_contours_by_x(contours):
    """
    Sort contours by the x coordinate of center point
    """

    num = len(contours)

    for i in np.arange(num - 1):
        for j in np.arange(i + 1, num):
            ci = contours[i]
            cj = contours[j]
            if ci.center_point.x > cj.center_point.x:
                contours[i] = cj
                contours[j] = ci

    return contours

def find_indexes_from_contours(contours):
    """
    Find contours form contours
    """

    indexes = []

    for contour in contours:
        indexes.append(contour.index)

    return indexes


def get_plate(targets, image_b):
    """
    Return the normalized license plate.
    targets: target objects.
    image_b: binary image.
    """


    target_num = len(targets)

    if target_num < 4:
        print("Error: target number is less than 4")
        sys.exit()


    cps = []
    for target in targets:
        cps.append(target.center_point)

    cp_a = cps[0]  # Fisrt center point
    cp_z = cps[-1] # Last center point

    # Width and height
    t_h = targets[0].height
    d = 0.3 * t_h
    d1 = 0.6 * t_h

    # Slope of center points
    slope = (cp_z.y - cp_a.y) / (cp_z.x - cp_a.x)

    point_l = find_point_along_line(cp_a, slope, d, -1)
    point_r = find_point_along_line(cp_z, slope, d, 1)

    # Slope of perpendicular line
    slope_p = -1.0 / slope

    # Vertices of the plate
    slope_p_sign = 1 if slope_p > 0 else -1
    vertex_ul = find_point_along_line(point_l, slope_p, d1, -slope_p_sign)
    vertex_ur = find_point_along_line(point_r, slope_p, d1, -slope_p_sign)
    vertex_lr = find_point_along_line(point_r, slope_p, d1, slope_p_sign)
    vertex_ll = find_point_along_line(point_l, slope_p, d1, slope_p_sign)

    vertices = []
    vertices.append(vertex_ul)
    vertices.append(vertex_ur)
    vertices.append(vertex_lr)
    vertices.append(vertex_ll)

    out = get_normalized_image(image_b, vertices)

    return out

def get_normalized_image(image, vertices, ny = 50, nx = 200):
    """
    Return the normalized images.
    """

    out = np.zeros((ny, nx), dtype = image.dtype)

    ul = vertices[0] # upper-left
    ur = vertices[1] # upper-right
    lr = vertices[2] # lower-right
    ll = vertices[3] # lower-left

    # Points of upper and lower lines
    points_upper = points_between_two_vertices(ul, ur, nx)
    points_lower = points_between_two_vertices(ll, lr, nx)

    # Points of left and right lines
    points_left = points_between_two_vertices(ul, ll, ny)
    points_right = points_between_two_vertices(ur, lr, ny)

    for iy in np.arange(ny):
        for ix in np.arange(nx):
            point = get_cross_point(points_left[iy], points_right[iy], points_upper[ix], points_lower[ix])
            xx = int(point.x + 0.5)
            yy = int(point.y + 0.5)
            out[iy, ix] = image[yy, xx]

    return out

def points_between_two_vertices(va, vz, n):
    """
    Return points between two vertices
    va, vz: the two vertices.
    n: number of points.
    """

    points = []
    for i in np.arange(n):
        x = linear_interp1(0, n-1, va.x, vz.x, i)
        y = linear_interp1(0, n-1, va.y, vz.y, i)
        points.append(Point(x, y))

    return points

def get_cross_point(p1, p2, p3, p4):
    """
    Return the cross point determined by two lines;
    where the first line passes throught p1 and p2 and
    the second line passes through p3 and p4.
    p1, p2, p3, p4: four points.

    """
    x1 = p1.x; y1 = p1.y
    x2 = p2.x; y2 = p2.y
    x3 = p3.x; y3 = p3.y
    x4 = p4.x; y4 = p4.y

    # First line y = a1*x + b1
    a1 = (y2 - y1) / (x2 - x1)
    b1 = -a1*x1 + y1

    # Second line y = a2*x + b2
    a2 = (y4 - y3) / (x4 - x3)
    b2 = -a2*x3 + y3

    # The cross point
    tmp = a1 - a2
    x = (-b1 + b2) / tmp
    y = (a1*b2 - a2*b1) / tmp
    cross_point = Point(x, y)

    return cross_point


def distance_2p(p1, p2):
    """
    Return the distance of two points.
    p1, p2: points.
    """
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    d = np.sqrt(dx*dx + dy*dy)

    return d

def get_char_contours(contours):
    """
    Return the character contours
    contours_in: input contours
    """

    # Filter the contour of dash
    contours = filter_dash(contours)

    n = len(contours)
    merged = np.zeros((n), dtype = np.bool)

    for i in np.arange(n):
        if merged[i]: continue
        ci = contours[i]
        ci_x_min = ci.x_min
        ci_x_max = ci.x_max
        for j in np.arange(n):
            if merged[j]: continue
            if j == i: continue
            cj = contours[j]
            cj_xc = cj.center_point.x
            merged[j] = check_bound1(cj_xc, ci_x_min, ci_x_max)
            if merged[j]: ci.merge(cj)

    # Pick the character contours
    out = []
    for i in np.arange(n):
        if not merged[i]: out.append(contours[i])

    # Sort contours by x coordinate
    out = sort_contours_by_x(out)

    return out

def get_normal_char_image(image, nx_out = 25, ny_out = 50):
    """
    Get the normalized character image
    image: input image
    """
    nxd2_out = int(nx_out / 2)

    ny, nx = image.shape

    nxd2 = int(nx / 2)
    nyd2 = int(ny / 2)
    ixc = nxd2

    if nx >= nx_out:
        ixl = ixc - nxd2_out
        ixh = ixc + nxd2_out
        out = image[:, ixl : ixh+1]
    else:
        out = image[:, :]

    return out

def cut_char_image(contour, image, nx_out = 25, ny_out = 50):
    """
    Cut character image
    """
    ny, nx = image.shape

    nxd2_out = int(nx_out / 2)
    ixc = int(contour.center_point.x)
    ixl = ixc - nxd2_out
    ixh = ixc + nxd2_out

    out = image[:, ixl: ixh+1]

    return out

def get_chars(image):
    """
    Get the characters of the license plate
    image: plate image
    """

    contours = get_contours(image)
    contours = get_char_contours(contours)

    n = len(contours)
    char_images = []
    for i in np.arange(n):
        contour = contours[i]
        x_min = int(contour.x_min + 0.5)
        x_max = int(contour.x_max + 0.5)
        contour_image = image[:, x_min : x_max+1]
        char_image = cut_char_image(contour, image)
        char_images.append(char_image)

    dash_index = get_dash_index(contours)
    out = char_images

    return out, dash_index

def match_chars(char1, char2):
    """
    Return the rate of matching two characters.
    """
    n = char1.size
    rate = 1.0*np.sum(char1 == char2) / n

    return rate

def filter_dash(contours, ny = 50):
    """
    Return the contours after filtering dash
    """

    threshold = 0.3*ny
    out = [c for c in contours if c.height > threshold]

    return out

def get_dash_index(contours):
    """
    Return the index of the dash symbol.
    """
    n = len(contours)
    index = -1
    d_max = 0
    for i in np.arange(n-1):
        c1_xc = contours[i].center_point.x
        c2_xc = contours[i+1].center_point.x
        d = c2_xc - c1_xc
        if d_max < d:
            d_max = d
            index = i

    index += 1

    return index

def predict_proba(X, coef):
    """
    Probability estimates.
    The returned estimates for all classes are ordered by the label of classes.
    """

    n_classes = coef.shape[0]

    if X.ndim == 1:
        n_samples = 1
    else:
        n_samples = X.shape[0]

    proba = np.zeros((n_samples, n_classes))
    for i in np.arange(n_samples):
        for j in np.arange(n_classes):
            x = X[i, :]
            proba[i, j] = logistic_predict_proba(x, coef[j, :])


    return proba

def get_license_numbers(char_images, dash_index, model):
    """
    Return license numbers.
    """

    char_num = len(char_images)

    numbers = []
    for i in np.arange(char_num):
        x = binary2gray(char_images[i]).flatten()
        y = model.predict(x)
        numbers.append(y[0])

    numbers.insert(dash_index, '-')
    numbers_str = list2str(numbers)

    return numbers_str

def recog(image, ml_labels, ml_coef):
    """
    Recognize the license plate from an image

    image : array
        image to be recognized.

    ml_labels: array-like, shape (n_classes)
        Class labels of machine learning

    ml_coef: array-like, shape (n_classes, n_features)
        Coefficients of machine learning
    """

    # Get the image with gray-scale
    ndim = image.ndim
    if ndim == 3:
        image_g = rgb2gray(image)
    else:
        image_g = image

    image_b = gray2binary(image_g)
    contours = get_contours(image_b)
    indexes = find_aligned_targets(contours)
    targets = find_contours_by_indexes(indexes, contours)
    targets = sort_contours_by_x(targets)

    plate_b = get_plate(targets, image_b)

    char_images, dash_index = get_chars(plate_b)
    model = PredictModel(ml_labels, ml_coef)

    license_numbers = get_license_numbers(char_images, dash_index, model)

    return license_numbers
