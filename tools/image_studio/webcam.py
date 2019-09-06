import cv2

class Webcam(object):

    def __init__(self):

        self.device = None # Device
        self.vc = None # Video capture
        self.frame = None

    def open(self, device, width=1920, height=1080, brightness=1, contrast=40, saturation=50,
             hue=50, exposure=50):
        # Set the target device.

        self.device = device
        self.vc = cv2.VideoCapture(self.device)

        # Set the camera parameters
        self.vc.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.vc.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        #self.vc.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
        #self.vc.set(cv2.CAP_PROP_CONTRAST, contrast)
        #self.vc.set(cv2.CAP_PROP_SATURATION, saturation)
        #self.vc.set(cv2.CAP_PROP_HUE, hue)
        #self.vc.set(cv2.CAP_PROP_EXPOSURE, exposure)

    def is_open(self):
        # Device is open or not.

        if self.vc:
            return self.vc.isOpened()
        else:
            return False


    def read(self):
        # Read the frame.

        is_capturing, frame = self.vc.read()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Save the frame
        self.frame = frame

        return frame

    def get_frame(self):
        return self.frame

    def release(self):
        # Release the resource.

        if self.is_open():
            self.vc.release()

