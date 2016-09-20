import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def random_noise(shape, amp=1):
    """"
    Return the random noise.

    Parameters:

    shape: array shape.
    amp: amplitude of noise.

    """     

    noise = amp * (2*np.random.rand(*shape) - 1)

    return noise

def get_timestamp(num_samples, it0=1000000, idt=10):
    """
    Return the fake timestamps.
    
    Parameters:
    
    num_samples: number of samples.
    it0: initial timestamp
    idt: integer time duration
    """ 
    
    timestamps = np.zeros(num_samples, dtype=np.int)
    
    for i in range(num_samples):
        timestamps[i] = it0 + i*idt
        
    return timestamps  


def get_joint_angles(num_samples, freq=0.1, dt=1):
    """
    Return the fake joint angles.
    
    theta = pi/6 * sin(2*pi*freq*t)
    
    Parameters
    
    num_samples: number of samples.
    freq: oscilation frequency (Hz).
    dt: time duration to pick a sample (second).
    """
   
    angles = np.zeros(num_samples)
        
    for i in range(num_samples):
        
        t = i*dt
        angles[i] = np.pi/6 * np.sin(2*np.pi*freq*t)
    
    return angles
    
    
def imp_from_angles(num_imp, angles):
    """
    Return the fake impedances calculated from joint angles.
    
    Parameters
    
    num_imp: number of impedances.
    angles: joint angles.
    """    
    
    num_samples = angles.size
    
    imp = np.zeros([num_samples, 2*num_imp])
    for i in range(num_imp):
        ii = 2*i
        imp[:, ii]   = i*angles # Real part
        imp[:, ii+1] = i*angles # imaginary part
 
    imp = imp / num_imp   
        
    return imp

def generate_wristband_file(filename="fake_wristband_data.csv", num_samples=10, num_imp=112):
    """
    Generate the fake data file for wristband.
    
    Parameters
    
    filename: filename.
    num_samples: number of samples.
    num_imp: number of impedances.
    """ 
    
    # Preparing the column names
    time_name = ["timestamp"]
    imp_names = []
    for i in range(num_imp):
        imp_names += ["imp_"+ str(i) + "_real"]
        imp_names += ["imp_"+ str(i) + "_img"]
    
    accel_names = ["accel_x", "accel_y", "accel_z"]
    gyro_names = ["gyro_x", "gyro_y", "gyro_z"]
    magneto_names = ["magneto_x", "magneto_y", "magneto_z"]
    ppg_name = ["ppg_value"]    
    
    
    # Dataframe of timestamps
    timestamp_array = get_timestamp(num_samples)
    timestamp_df = pd.DataFrame(timestamp_array, columns=time_name) 
    
    # Dataframe of impedances
    angles = get_joint_angles(num_samples)
    imp_array = imp_from_angles(num_imp, angles)
    ## Add random noise
    noise_amp = imp_array.max()*0.05
    imp_array += random_noise(imp_array.shape, amp=noise_amp)
    imp_df = pd.DataFrame(imp_array, columns=imp_names)
    
    # Dataframe of accelerometer
    accel_array = np.zeros((num_samples, 3))
    accel_df = pd.DataFrame(accel_array, columns=accel_names)
    
    # Dataframe of gyrometer
    gyro_array = np.zeros((num_samples, 3))
    gyro_df = pd.DataFrame(gyro_array, columns=gyro_names)
    
    # Dataframe of magnetometer
    magneto_array = np.zeros((num_samples, 3))
    magneto_df = pd.DataFrame(magneto_array, columns=magneto_names)
    
    # Dataframe of PPG
    ppg_array = np.zeros((num_samples))
    ppg_df = pd.DataFrame(ppg_array, columns=ppg_name)
    
    # Final dataframe
    df = pd.concat([timestamp_df, imp_df, accel_df, gyro_df, magneto_df, ppg_df], axis=1)
    
    # Output as CSV file
    print("Output {}".format(filename))
    df.to_csv(filename, index=False)
    
    return df
    
def generate_glove_file(filename="fake_glove_data.csv", num_samples=10, num_joints=15):
    """
    Generate the fake data file for glove.
    
    Parameters
    
    filename: filename.
    num_samples: number of samples.
    """ 
    
    # Preparing the column names
    time_name = ["timestamp"]

    joint_names = []
    for i in range(num_joints):
        joint_names += ["joint_" + str(i)]
    
    accel_names = ["accel_x", "accel_y", "accel_z"]
    gyro_names = ["gyro_x", "gyro_y", "gyro_z"]
    magneto_names = ["magneto_x", "magneto_y", "magneto_z"]
    
    # Dataframe of timestamps
    timestamp_array = get_timestamp(num_samples)
    timestamp_df = pd.DataFrame(timestamp_array, columns=time_name) 
    
    # Dataframe of joints
    joint_array = np.zeros((num_samples, num_joints))
    for i in range(num_joints):
        joint_array[:, i] = get_joint_angles(num_samples)
    ## Add random noise
    noise_amp = joint_array.max()*0.05

    joint_array += random_noise(joint_array.shape, amp=noise_amp)
    joint_df = pd.DataFrame(joint_array, columns=joint_names)
    
    # Dataframe of accelerometer
    accel_array = np.zeros((num_samples, 3))
    accel_df = pd.DataFrame(accel_array, columns=accel_names)
    
    # Dataframe of gyrometer
    gyro_array = np.zeros((num_samples, 3))
    gyro_df = pd.DataFrame(gyro_array, columns=gyro_names)
    
    # Dataframe of magnetometer
    magneto_array = np.zeros((num_samples, 3))
    magneto_df = pd.DataFrame(magneto_array, columns=magneto_names)
    
    # Final dataframe
    df = pd.concat([timestamp_df, joint_df, accel_df, gyro_df, magneto_df], axis=1)
    
    # Output as CSV file
    print("Output {}".format(filename))
    df.to_csv(filename, index=False)
    
    return df
    

class Faker:
    """
    Class of fake data generator which can generate the raw data collected from devices 
    for model training.
    """
      
    def __init__(self, num_samples=10, num_joints=15, num_imp=4):
        """
        Class constructor
        """
        self._num_samples = num_samples
        self._num_joints = num_joints   
        self._num_imp = num_imp   
            
    def get_num_samples(self):
        """
        Return the number of samples
        """
        return self._num_samples
    
    def get_num_joints(self):
        """
        Return the number of joints
        """
        return self._num_joints

    def get_num_imp(self):
        """
        Return the number of impedances (complex values)
        """
        return self._num_imp
    
    
    def show_key_info(self):
        """
        Show the key information.
        """
        print("---- Key information ----")
        print("Number of samples: {}".format(self.get_num_samples()))
        print("Number of joints: {}".format(self.get_num_joints()))
        print("Number of complex impedances: {}".format(self.get_num_imp()))
        print("-------------------------------")
        
    
    def output_wristband_file(self, filename="fake_wristband_data.csv"):
        """ 
        Output the data file for wristband and return the dataframe.
    
        Parameters
    
        filename: filename.
        """
        num_samples=self.get_num_samples()
        num_imp = self.get_num_imp()
        df = generate_wristband_file(filename=filename, \
            num_samples=num_samples, num_imp=num_imp) 
     
        return df 

    def output_glove_file(self, filename="fake_glove_data.csv"):
        """ 
        Output the data file for glove and return the dataframe.
    
        Parameters
    
        filename: filename.
        """
        num_samples = self.get_num_samples()
        num_joints = self.get_num_joints()
        df = generate_glove_file(filename=filename, \
            num_samples=num_samples, num_joints=num_joints)
     
        return df

