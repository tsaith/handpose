
class FingerModel:
    """
    Finger model class.

    """

    def __init__(self, joints):
        """
        Constructor of finger model.

        Parameters
        ----------
        joints: list
            Joints; three elements are expected. 

        """
        pass 


    def set_joints(self, joints):
        """ 
        Set the joints.

        Parameters
        ----------
        joints: array-like
            Joints; three elements are expected. 

        """
        pass

    @property
    def joints(self):
        """ 
        Return the joints.
        """
        pass

    def set_phalanges(self, phalanges):
        """ 
        Set the phalanges.

        Parameters
        ----------
	phalanges: list
            Phalanges.

        """
        pass

    @property
    def phalanges(self):
        """ 
        Return the phalanges.
        """
        pass

