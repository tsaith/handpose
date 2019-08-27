import numpy as np

def get_distance_from_two_points(x1, y1, x2, y2):

    dx = x2-x1
    dy = y2-y1
    d = np.sqrt(dx*dx+dy*dy)

    return d

def get_distance_point_line(x0, y0, a, b, c):
    # Get the distance from a point(x0, y0) to a line (a*x+b*y+c = 0).

    d = ( a*x0 + b*y0 + c ) / np.sqrt(a*a + b*b)

    return d

def get_perp_foot(x0, y0, a, b, c):
    # Get the perpendicular foot when considering a point and a line.
    # The point is (x0, y0) and the line is a*x+b*y+c = 0.

    x = (b*b*x0 - a*b*y0 - a*c) / (a*a + b*b)
    y = (-a*b*x0 + a*a*y0 - b*c) / (a*a + b*b)

    return x, y

def get_line_from_two_points(x1, y1, x2, y2):
    # Get the coefficients of a line: a*x+b*y+c = 0.

    a = (y2-y1) / (x2-x1)
    b = -1
    c = -a*x1 + y1

    return a, b, c

def get_perp_foot_from_three_points(x0, y0, x1, y1, x2, y2):
    # Get the perpendicular foot from three points.
    # The p0 is (x0, y0).
    # The p1 and p2 form a line.

    # Line
    a, b, c = get_line_from_two_points(x1, y1, x2, y2)

    # Perpendicular foot
    xp, yp = get_perp_foot(x0, y0, a, b, c)

    return xp, yp
