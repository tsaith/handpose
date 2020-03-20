

class BodyDetector:
    '''
    Body detector.
    '''

    def __init__(self, engine_type='openpose', model_dir=None):

        self.engine_type = engine_type
        self.model_dir = model_dir

        self.input_image = None
        self.output_image = None
        self.keypoints = None

        self.datum = None
        self.opWrapper = None

        if engine_type == 'openpose':

            # Import module
            from openpose import pyopenpose as op

            # Initialize OpenPose

            # Custom Params (refer to include/openpose/flags.hpp for more parameters)
            params = dict()
            params["model_folder"] = self.model_dir
            self.opWrapper = op.WrapperPython()
            self.opWrapper.configure(params)
            self.opWrapper.start()

            self.datum = op.Datum()
        else:

            # Import module
            import trt_pose as tp

    def set_input_image(self, image):
        self.input_image = image.copy()

    def get_output_image(self):
        return self.output_image

    def get_keypoints(self):
        return self.keypoints

    def process(self):

        # Obtain key points
        if self.engine_type == 'openpose':
            datum = self.datum
            datum.cvInputData = self.input_image
            self.opWrapper.emplaceAndPop([datum])
            self.keypoints = datum.poseKeypoints
            self.output_image = datum.cvOutputData.copy()

        else:
            pass

        return self.keypoints
