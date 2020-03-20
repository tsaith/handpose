import numpy as np
import cv2


class EgoTracker:
    """
    Ego tracker.
    """

    def __init__(self, frame, bbox, tracker_type='Farneback'):
        """
        Constructor.

        Parameters
        ----------
        frame : array
            Video frame.
        tracker_type: string
            Tracker type; cadidates: 'Farneback', 'LK'
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

        # Tracker type
        self._tracker_type = tracker_type

        # Velocity
        self._vx = None # In terms of pixel
        self._vy = None

        # Displacement
        self._dx = None # In terms of pixel
        self._dy = None

        # Previous frame
        self._frame_prev = frame

        # Optical flow
        self._flow = None

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
    def flow(self):
        """
        Return the flow vectors.
        """
        return self._flow

    @property
    def bbox(self):
        """
        Return the updated bbox.
        """
        return self._bbox

    def update(self, frame):
        """
        Update the tracking.
        """

        # Bounding box
        bbox = self._bbox_ori
        x_a = bbox[0]
        y_a = bbox[1]
        x_z = bbox[0]+bbox[2]
        y_z = bbox[1]+bbox[3]

        # Previous and current frame
        prev = self._frame_prev[x_a:x_z, y_a:y_z]
        curr = frame[x_a:x_z, y_a:y_z]

        # Estimate the optical flow
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.5, 0)
        dx = flow[..., 0]
        dy = flow[..., 1]

        # Mean dx and dy
        dx_mean = np.mean(dx)
        dy_mean = np.mean(dy)

        # Ego displacement
        self._dx = -dx_mean
        self._dy = -dy_mean

        # Update the previous frame
        self._frame_prev = frame

        # Save the flow fields
        self._flow = flow

    def is_motional(self, shift_th=0.2):
        """
        Judge if motional or not.

        shift_th: float
            Threshold of the shift.
        """
        shift = np.sqrt(self.dx*self.dx + self.dy*self.dy)
        if shift > shift_th:
            motional = True
        else:
            motional = False

        return motional


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

