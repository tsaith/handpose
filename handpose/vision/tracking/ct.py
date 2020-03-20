import numpy as np
import sys
import cv2

from ..lib.libcore import Rect
from ..core import rand_uniform, floor, ceil
from ..gui import wait_key
from ..draw import draw_rect

from ..lib.libct import CyCompressiveTracker as CompressiveTracker
from ..lib.libct import cy_get_feature_values, cy_ratio_classifier

class PyCompressiveTracker():
    """
    Compressive tracker.
    """

    def __init__(self, frame, object_box, n_features=50, learn_rate=0.85):

        self._n_features = n_features

        self._radius_pos = 4 # Radius of positive sampling
        self._radius_search = 25 # Radios of search window

        self._mu_pos = np.zeros(n_features)
        self._mu_neg = np.zeros(n_features)
        self._sigma_pos = np.ones(n_features)
        self._sigma_neg= np.ones(n_features)
        self._learn_rate = learn_rate # Learning rate

        # --------
        # Harr-like features
        self._features, self._feature_weights = self.haar_features(object_box, self._n_features)

        # Update distribution
        self.update_distr(frame, object_box)

    def process_frame(self, frame, object_box):
        """
        Process the frame.
        """

        # Update the object box
        detect_boxes = self.sample_boxes(frame, object_box, self._radius_search, 0, 1000000)
        self._integral_image = cv2.integral(frame, sdepth=cv2.CV_64F)
        feature_values = cy_get_feature_values(detect_boxes, self._features, self._feature_weights, self._integral_image)

        index_ratio_max, _ = cy_ratio_classifier(self._mu_pos, self._sigma_pos, self._mu_neg, self._sigma_neg, feature_values)
        object_box = detect_boxes[index_ratio_max]

        # Update distribution
        self.update_distr(frame, object_box)

        return object_box

    def update_distr(self, frame, object_box):

        # Obtain the positive and negative samples
        boxes_pos = self.sample_boxes(frame, object_box, self._radius_pos, 0, 1000000)
        boxes_neg = self.sample_boxes(frame, object_box, self._radius_search*1.5, self._radius_pos+4, 100)

        # Integral image
        integral_image = cv2.integral(frame, sdepth=cv2.CV_64F)

        feature_values_pos = cy_get_feature_values(boxes_pos, self._features, self._feature_weights, integral_image)
        feature_values_neg = cy_get_feature_values(boxes_neg, self._features, self._feature_weights, integral_image)

        self.update_classifier(self._mu_pos, self._sigma_pos, feature_values_pos, self._learn_rate)
        self.update_classifier(self._mu_neg, self._sigma_neg, feature_values_neg, self._learn_rate)

    def sample_boxes(self, image, object_box, outer_radius, inner_radius, n_samples_max):
        """
        Return sampling boxes.
        Samples lie between outer and inner circles.

        Parameters
        ---------
        image : array-like, shape(rows, cols)
            Image with grayscale
        outer_radius : scalar
            Radius of the outer circle.
        inner_radius : scalar
            Radius of the inner circle.
        n_samples_max : integer
            Maximal number of samples.

        Return
        ------
        boxes : object array.
            Sampling boxes.
        """

        box_x = object_box.x
        box_y = object_box.y
        box_w = object_box.width
        box_h = object_box.height

        box_rows = box_h + 1
        box_cols = box_w + 1

        image_rows, image_cols = image.shape
        rows_z = image_rows - box_rows
        cols_z = image_cols - box_cols

        # Ends of outer sphere
        x_min = int(box_x - outer_radius)
        x_max = int(box_x + outer_radius)
        y_min = int(box_y - outer_radius)
        y_max = int(box_y + outer_radius)

        x_min = np.max([0, x_min])
        x_max = np.min([cols_z - 1, x_max])
        y_min = np.max([0, y_min])
        y_max = np.min([rows_z - 1, y_max])

        # Probability
        n_outer = (x_max - x_min + 1)*(y_max - y_min + 1)
        proba = 1.0 * n_samples_max / n_outer

        r2_inner = inner_radius * inner_radius
        r2_outer = outer_radius * outer_radius

        boxes = []
        for y in np.arange(y_min, y_max + 1):
            for x in np.arange(x_min, x_max + 1):
                # Square of distance
                d2 = (x - box_x)*(x - box_x) + (y - box_y)*(y - box_y)
                proba_rand = np.random.rand()
                if proba_rand < proba and d2 >= r2_inner and d2 < r2_outer:
                    box = Rect(x, y, box_w, box_h)
                    boxes.append(box)

        boxes = np.array(boxes, dtype=object)

        return boxes

    def update_classifier(self, mu, sigma, feature_values, learn_rate):
        """
        Update the mean and standard deviation of the classifier.

        Parameters
        ----------
        mu : array-like, shape (n_features)
            The mean values.

        sigma : array-like, shape (n_features)
            The standard deviations.

        feature_values : array-like, shape (n_features, n_samples)
            Feature values.

        learn_rate : scalar
            Learning rate.
        """

        one_m_learn_rate = 1.0 - learn_rate

        for i in np.arange(self._n_features):
            mu_tmp, sigma_tmp = cv2.meanStdDev(feature_values[i, :])
            mu[i] = learn_rate*mu[i] + one_m_learn_rate*mu_tmp
            sigma[i] = np.sqrt(
                learn_rate*sigma[i]*sigma[i] +
                one_m_learn_rate*sigma_tmp*sigma_tmp +
                learn_rate*one_m_learn_rate*(sigma[i]-sigma_tmp)*(sigma[i]-sigma_tmp))

    def ratio_classifer(self, feature_values):
        """
        Ratio classifier.

        Return
        ------
        index_ratio_max : integer.
            The index of maximal ratio.

        ratio_max : float.
            The value of maximal ratio.
        """

        err_tol = 1.0e-30 # error tolerance

        mu_pos = self._mu_pos
        mu_neg = self._mu_neg

        sigma_pos = self._sigma_pos
        sigma_neg = self._sigma_neg

        ratio_max = np.finfo('float').min
        index_ratio_max = 0

        n_features, n_samples = feature_values.shape

        for j in np.arange(n_samples):
            rsum = 0.0
            for i in np.arange(n_features):
                v = feature_values[i, j]

                fac = 1.0 / (sigma_pos[i] + err_tol)
                proba_pos = fac * np.exp(-0.5 * (v - mu_pos[i])*(v - mu_pos[i]) / (sigma_pos[i]*sigma_pos[i] + err_tol))

                fac = 1.0 / (sigma_neg[i] + err_tol)
                proba_neg = fac * np.exp(-0.5 * (v - mu_neg[i])*(v - mu_neg[i]) / (sigma_neg[i]*sigma_neg[i] + err_tol))

                rsum += np.log(proba_pos + err_tol) - np.log(proba_neg + err_tol)

            if (ratio_max < rsum):
                ratio_max = rsum
                index_ratio_max = j

        return index_ratio_max, ratio_max


    def haar_features(self, box, n_features, n_rects_min=2, n_rects_max=4):
        """
        Calculate the Harr-like features.

        Parameters
        ----------
        object_box : object.
            Rectangular object box.

        n_features : integer.
            Number of features.
        """

        features = np.empty((n_features, n_rects_max), dtype=object)
        weights = np.zeros((n_features, n_rects_max), dtype=np.float64)
        n_rects = np.zeros((n_features), dtype=np.int32)

        box_width  = box.width
        box_height = box.height

        rect0 = Rect(0, 0, 0, 0)
        features[:, :] = rect0

        for i in range(n_features):
            n_rects[i] = np.int(np.floor(rand_uniform(n_rects_min, n_rects_max))) # random integer of [low, high), this may have a bug?
            for j in range(n_rects[i]):
                x = floor(rand_uniform(0.0, box_width - 3))
                y = floor(rand_uniform(0.0, box_height - 3))
                width  = ceil(rand_uniform(0.0, box_width - x - 2))
                height = ceil(rand_uniform(0.0, box_height - y - 2))
                rect = Rect(x, y, width, height)
                features[i, j] = rect

                sign = np.power(-1.0, floor(rand_uniform(0.0, 2.0)))
                weights[i, j] = sign/np.sqrt(n_rects[i])

        return features, weights

def detect_box(cap, win_name):
    """
    Define a detection box.

    Parameters
    ----------
    cap: object.
        video capture.
    win_name : String.
        Window name.
    """

    box_defined = False
    box = Rect(0, 0, 0, 0)

    def define_box(event, x, y, flags, param):

        nonlocal box_defined, box
        if event == cv2.EVENT_LBUTTONDOWN:
            box.x = x
            box.y = y
            box.width  = 0
            box.height = 0

        if event == cv2.EVENT_MOUSEMOVE:
            box.width  = x - box.x
            box.height = y - box.y

        if event == cv2.EVENT_LBUTTONUP:
            box_defined = True

    def do_nothing(event, x, y, flags, param):
        pass

    # set mouse callback
    cv2.setMouseCallback(win_name, define_box)

    while not box_defined:
        # display the frame from video capture
        _, frame = cap.read()
        clone = frame.copy()
        if box.x > 0 and box.width > 0:
            draw_rect(clone, box, (0, 255, 0), 2)
        cv2.imshow(win_name, clone)

        key = wait_key(10) # This is important for activating the mouse callback

    # Set a mouse callback which does nothing
    cv2.setMouseCallback(win_name, do_nothing)

    return box, frame
