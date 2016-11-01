import numpy as np
import pandas as pd
import os

def find_class_files(class_label, dir_path):
    """
    Find out the files matching the specific class label. 
    
    Parameters
    ----------
    class_label: str
        The class label.
        
    dir_path: str
        The directory path.
    
    Returns
    -------
    filenames: list
        The list of filenames.
    """   

    ext = "csv"
    
    filenames = []
    for f in os.listdir(dir_path):
        if f.startswith(class_label) and f.endswith(ext):
            filenames.append(f)

    return filenames       

def csv2numpy(file_path):
    """
    Load a CSV file and convert its content into a numpy array.
    
    Parameters
    ----------
    file_path: str
        Path of the CSV file.
    
    Returns
    -------
    arr: numpy array
        Data array.
    """  

    # Read the CSV file as dataframe
    df  = pd.read_csv(file_path)
    
    # Skip the timestamp
    #df = df.iloc[:, 1:] # skip the timestamp

    # Convert dataframe into numpy array 
    arr = df.values

    # Set the data type as float
    arr = arr.astype(np.float32)
    
    return arr
    
def load_class_data(label_candidates, dir_path):
    """
    Load the training data for classification.
    
    Parameters
    ----------
    label_candidates: list
        Label candidates.
    
    dir_path: str
        Directory path.
    
    Returns
    -------
    data: array-like
        Training data.
    """   

    num_classes = len(label_candidates)
    
    class_files =  [[] for c in range(num_classes)]
    class_data =   [[] for c in range(num_classes)]
    class_labels = [[] for c in range(num_classes)]
  
    for c in range(num_classes): 
        filenames = find_class_files(label_candidates[c], dir_path)
        class_files[c].append(filenames)
        
        # Prepare class data
        for filename in filenames: 
            file_path = "{}/{}".format(dir_path, filename)
            print("Reading the file: {}".format(file_path))
            arr = csv2numpy(file_path)
            arr = arr[:, 1:] # Remove the timestamp
            print("The shape of data is {}".format(arr.shape))
            class_data[c].append(arr)

        class_data[c] = np.concatenate(class_data[c]) 
 
        # Prepare the class labels
        class_labels[c] = np.full(len(class_data[c]), c, dtype=np.int32)

        print("--------")
        print("There are {} instances in class {}".format(len(class_labels[c]), c))
        print("--------")

    # Truncate instances for equal weighting
    num_min_instances = 100000000
    for c in range(num_classes):
        num_instances = len(class_data[c])
        if num_min_instances < num_instances:
            num_min_instances = num_instances
    
    # Concatenate data and labels
    data = np.concatenate(class_data) 
    labels = np.concatenate(class_labels)
        
    return data, labels

