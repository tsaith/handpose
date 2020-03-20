import numpy as np

def get_angle(x, y):
    return np.arctan2(-y, x)


class WristGesture:
    '''
    Wrist gesture.
    '''

    def __init__(self, num_points_max=3):

        self.num_points_max = num_points_max
        self.dist_c = 100

        self.points = []

    def set_dist_c(self, d):
        self.dist_c = d

    def update(self, point):
        # Update data.

        self.points.append(point)

        # Pop the oldest point.
        if len(self.points) > self.num_points_max:
            self.points.pop(0)

    def recognize(self):

        points = np.array(self.points)

        num_points = len(points)

        p0 = self.points[0][0:2]
        p1 = self.points[num_points-1][0:2]
        dv = p1 - p0
        d = np.linalg.norm(dv)
        dx, dy = dv[0], dv[1]
        angle = get_angle(dx, dy)

        result = None
        abs_angle = abs(angle)
        if d > self.dist_c:
            if abs_angle < np.pi/6:
                result = "right"

            if abs_angle > np.pi*5/6:
                result = "left"

            if abs_angle > np.pi*2/6 and abs_angle < np.pi*4/6:
                if dy > 0:
                    result = "down"
                else:
                    result = "up"

        return result
