import keras


class GestureEngine:
    """
    Gesture engine class.

    """

    def __init__(self, config):
        """
        Constructor of recognition engine.
    
        Parameters
        ----------
        config : object
            Configuration containing the information of system, 
            pre-processing and post-processing.

        """
        self.model_trained = None 


    def set_config(self, config):
        """ 
        Set the configuration.

        Parameters
        ----------
	config: object
            Configuration object.

        """
        pass

    
    @property
    def config(self):
        """
        Return the configuration.
        """
        pass

    def set_wristband(self, wristband):
        """
        Set the wristband.

        Parameters
        ----------
	wristband: object
            Wristband object.
        """
        pass

    @property
    def wristband(self):
        """
        Return the wristband.
        """
        pass

    def set_glove(self, glove):
        """
        Set the glove.

        Parameters
        ----------
	glove: object
            glove object.
        """
        pass

    @property
    def glove(self):
        """
        Return the glove.
        """
        pass

    def load_model(self, file_path):
        """
        Load the traind model file.

        Parameters
        ----------
	file_path: str
            File path to the trained model.
        """
        self.model_trained = keras.models.load_model(file_path)

        return self.model_trained
        

    def predict_classes(self, x):
        """
        Generate class predictions for the input samples.

        Parameters
        ----------
	X: array-like
            Input features.
        """
        y = self.model_trained.predict_classes(x, batch_size=32, verbose=0)
        
        return y


    def predict(self, x):
        """
        Generates output predictions for the input samples.

        Parameters
        ----------
	x: array-like
            Input features.
        """
        y = self.model_trained.predict(x, batch_size=32, verbose=0)

        return y

    def diagnostic(self):
        """
        Return the diagnostic information.

        """
        pass



class EngineConfig:
    """
    Engine configuration class.

    """

    def __init__(self):
        """
        Constructor of engine configuration.
    
        """
        pass 



    def set_dt(self, dt):
        """ 
        Set the time duration.

        Parameters
        ----------
	dt: float
            Time duration in milliseconds.

        """
        pass

