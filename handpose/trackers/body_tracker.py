import numpy as np
from touch.trackers import PointTracker

class BodyTracker:
    '''
    Body tracker.
    '''

    def __init__(self):

        self.target_keypoints = None
        self.num_people = 0
        self.target_keypoints = None

        # Point tracker
        self.point_tracker = PointTracker()

    def update(self, keypoints):

        try:
            assert len(keypoints) > 0
        except:
            return None

        # Noses
        noses = []
        for i, kp in enumerate(keypoints):
           noses.append(kp[0][0:2])

        self.point_tracker.update(noses)
        ref_point = self.point_tracker.get_target_position()

        # Target index
        target_index = None
        d_min = 10e8
        for i, nose in enumerate(noses):
            x = nose[0]
            y = nose[1]
            d = np.linalg.norm((x-ref_point[0], y-ref_point[1]))

            if d < d_min:
                target_index = i
                d_min = d

        # Target keypoints
        if target_index is None:
            self.target_keypoints = None
        else:
            self.target_keypoints = keypoints[target_index]

        return self.target_keypoints

    def get_target_body(self):
        return self.target_keypoints
