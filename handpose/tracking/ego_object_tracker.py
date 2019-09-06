import numpy as np
import cv2


class EgoObjectTracker:
    """
    Ego tracker 2d.
    """

    def __init__(self, frame, bbox, tracker_type='MEDIANFLOW'):
        """
        Constructor.

        Parameters
        ----------
        frame : array
            Video frame.
        tracker_type: string
            Tracker type; cadidates: 'MEDIANFLOW', 'KCF'
        """
        # Video frame
        if frame.ndim == 3:
            height, width, channels = frame.shape
        else:
            height, width = frame.shape
            channels = None

        self._frame_width = width
        self._frame_height = height
        self._frame_channels = channels

        # Bounding box
        self._bbox_ori = bbox
        self._bbox = None

        # Displacement
        self._dx = None # In terms of pixel
        self._dy = None

        # Define tracker
        if tracker_type == 'BOOSTING':
            self._tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            self._tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            self._tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            self._tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            self._tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            self._tracker = cv2.TrackerGOTURN_create()

        # Previous frame
        self._frame_prev = frame

    @property
    def dx(self):
        """
        Return x displacement in terms of pixel
        """
        return self._dx

    @property
    def dy(self):
        """
        Return y displacement in terms of pixel
        """
        return self._dy

    @property
    def bbox(self):
        """
        Return the updated bbox.
        """
        return self._bbox

    @property
    def tracker(self):
        """
        Return x displacement in terms of pixel
        """
        return self._tracker

    def update(self, frame):
        """
        Update the tracking.
        """

        # Coordinates of original bbox
        bbox = self._bbox_ori
        x0 = bbox[0]
        y0 = bbox[1]

        #self._tracker = cv2.TrackerBoosting_create()
        self._tracker = cv2.TrackerMIL_create()
        #self._tracker = cv2.TrackerKCF_create()
        #self._tracker = cv2.TrackerMedianFlow_create()
        #self._tracker = cv2.TrackerTLD_create()

        self.tracker.init(self._frame_prev, bbox)

        ok, bbox = self.tracker.update(frame)

        # Coordinates of updated bbox
        x = bbox[0]
        y = bbox[1]

        # Ego displacement
        self._dx = x0 - x
        self._dy = y0 - y

        # Save the updated bbox
        self._bbox = bbox

        # Update the previous frame
        self._frame_prev = frame

        return ok, bbox

    def is_motional(self, delta_th=3):
        """
        Judge if motional or not.

        delta_th: float
            Threshold of the displacement.
        """
        delta = np.sqrt(self.dx*self.dx + self.dy*self.dy)
        if delta > delta_th:
            motional = True
        else:
            motional = False

        return motional

