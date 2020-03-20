import numpy as np


def get_distance(x0, y0, x1, y1):

    dx = x1 - x0
    dy = y1 - y0
    out = np.sqrt(dx*dx+dy*dy)

    return out

class Anchor:

    def __init__(self, count_max=5):

        self.count_max = count_max
        self.count = 0
        self.point = None
        self.len_c = 100 # Critical length

    def update(self, point):

        if self.point is None:
            self.point = point

        (x, y) = point
        (x0, y0) = self.point
        d = get_distance(x0, y0, x, y)

        if d < self.len_c:
            self.count += 1
        else:
            self.count = 0

        if self.count >= self.count_max:
            self.point = point
            self.count = 0

        return self.point

class ObjectTracker:

    def __init__(self, width, height):

        # Frame resolution
        self.width = width
        self.height = height

        # Target
        self.has_target = False
        self.target = None
        self.target_prev = None
        self.anchor = Anchor()

    def get_target(self, objects):

        if len(objects) > 0: # Has a target
            d_min = 100000000.0
            if self.target_prev is not None: # Has a previous target
                (x_anchor, y_anchor) = self.anchor.update(self.target_prev[0:2])
                for obj in objects:
                    x = obj[0]
                    y = obj[1]
                    d = get_distance(x_anchor, y_anchor, x, y)

                    if d < d_min:
                        d_min = d
                        self.target = obj


            else: # No previous target
                self.target = objects[0]

        else: # No target
            self.target = None

        # Save as the previous target
        self.target_prev = self.target

        return self.target

    def get_position(self):
        # Return the normalized position.

        if self.target is not None:
            (x, y, w, h) = self.target
            xc = x + 0.5*w
            yc = y + 0.5*h
            x_norm = 1.0 * xc / self.width
            y_norm = 1.0 * yc / self.height
        else:
            x_norm = None
            y_norm = None

        return (x_norm, y_norm)
