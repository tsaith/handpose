
class BodyModel:
    '''
    Body_25 model.
    '''

    def __init__(self):

        self.keypoints = None
        self.has_body = False

    def set_keypoints(self, keypoints):
        self.keypoints = keypoints
        try:
            if len(keypoints) > 0:
                self.has_body = True
        except:
            self.has_body = False

    def get_nose(self):

        result = None
        if self.has_body:
            result = self.keypoints[0]

        return result

    def get_neck(self):

        result = None
        if self.has_body:
            result = self.keypoints[1]

        return result

    def get_rightwrist(self):

        result = None
        if self.has_body:
            result = self.keypoints[4]

        return result

    def get_leftwrist(self):

        result = None
        if self.has_body:
            result = self.keypoints[7]

        return result

    def get_midhip(self):

        result = None
        if self.has_body:
            result = self.keypoints[8]

        return result

    def get_rightwrist_mirrored(self):
        return self.get_leftwrist()

    def get_leftwrist_mirrored(self):
        return self.get_rightwrist()
