import cv2

class Webcam(object):

    def __init__(self):

        self.device = None # Device
        self.vc = None # Video capture
        self.frame = None

    def open(self, device):
        # Set the target device.

        self.device = device
        self.vc = cv2.VideoCapture(self.device)

    def is_open(self):
        # Device is open or not.

        if self.vc:
            return self.vc.isOpened()
        else:
            return False


    def read(self):
        # Read the frame.

        is_capturing, frame = self.vc.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Save the frame
        self.frame = frame

        return frame

    def get_frame(self):
        return self.frame

    def release(self):
        # Release the resource.

        if self.is_open():
            self.vc.release()

