import keras
import cPickle as pickle

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
        self.config = config
        self.scaler = None 
        self.model_trained = None 

        # Load the scaler
        self.load_scaler()

        # Load the trained model
        self.load_model()

    def set_config(self, config):
        """ 
        Set the configuration.

        Parameters
        ----------
	config: object
            Configuration object.

        """
        self.config = config

    
    @property
    def config(self):
        """
        Return the configuration.
        """
        return self.config

    def load_scaler(self, scaler_path=None):
        """
        Load the scaler file.

        Parameters
        ----------
	scaler_path: str
            The scaler path.
        """

        if scaler_path == None:
            scaler_path = self.config.scaler_path

        self.scaler = pickle.load(open(scaler_path, "rb"))

        return self.scaler
        

    def load_model(self, model_path=None):
        """
        Load the traind model file.

        Parameters
        ----------
	model_path: str
            The model path.
        """
        if model_path == None:
            model_path = self.config.model_path

        self.model_trained = keras.models.load_model(model_path)

        return self.model_trained
        

    def predict_classes(self, x):
        """
        Generate class predictions for the input samples.

        Parameters
        ----------
	X: array-like
            Input features.
        """

        x = self.scaler.transform(x)

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

