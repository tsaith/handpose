import numpy as np
import pandas as pd
import os

def csv2numpy(file_path, start_col=0, header='infer'):
    """
    Load a CSV file and output as a numpy array.

    Parameters
    ----------
    file_path: str
        Path of the CSV file.
    start_col: int
        Starting column index.
    header: 'infer', None, list
        Header.
    Returns
    -------
    arr: numpy array
        Data array.
    """

    # Read the CSV file as dataframe
    df  = pd.read_csv(file_path, header=header)
    arr = df.values

    # Set the data type as float
    arr = arr.astype(np.float32)

    # Skip the timestamp
    arr = arr[:, start_col:]

    return arr

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

def load_class_data(candidates, dir_path, 
    num_cols=None, equal_weight=True, start_col=0, header='infer', verbose=0):
    """
    Load the training data for classification.

    Parameters
    ----------
    candidates: list
        Class candidates.
    dir_path: str
        Directory path.
    num_cols: int
        Number of data colums.
    equal_weight: bool
        Equal weight for each class.
    verbose: int
        Switch used to show the debug information.
        Available value is 0 or 1.

    Returns
    -------
    data: array-like
        Training data.
    """

    num_classes = len(candidates)

    class_files =  [[] for c in range(num_classes)]
    class_data =   [[] for c in range(num_classes)]
    class_labels = [[] for c in range(num_classes)]

    for c in range(num_classes):
        filenames = find_class_files(candidates[c], dir_path)
        class_files[c].append(filenames)

        # Prepare class data
        for filename in filenames:
            file_path = "{}/{}".format(dir_path, filename)
            if verbose > 0:
                print("Reading the file: {}".format(file_path))
            arr = csv2numpy(file_path, start_col=start_col, header=header)
            arr = arr[:, start_col:] # Remove the timestamp
            if verbose > 0:
                print("The shape of data is {}".format(arr.shape))

            if num_cols == None:
                num_cols = arr.shape[1]
            class_data[c].append(arr[:, :num_cols])

        class_data[c] = np.concatenate(class_data[c])

        # Prepare the class labels
        class_labels[c] = np.full(len(class_data[c]), c, dtype=np.int32)

        if verbose > 0:
            print("--------")
            print("There are {} instances in class {} (labeled as {})".format(len(class_labels[c]), candidates[c], c))

    # Truncate instances for equal weighting
    if equal_weight:
        num_min = 100000000
        for c in range(num_classes):
            num_instances = len(class_data[c])
            if num_min > num_instances:
                num_min = num_instances

        for c in range(num_classes):
            class_data[c] = class_data[c][:num_min, :]
            class_labels[c] = class_labels[c][:num_min]

        if verbose > 0:
            print("--------")
            print("For equal weight, each class uses {} instances".format(num_min))

    # Concatenate data and labels
    data = np.concatenate(class_data)
    labels = np.concatenate(class_labels)

    if verbose > 0:
        num_instances, num_features = data.shape
        print("--------")
        print("Instance number is {}".format(num_instances))
        print("Feature number is {}".format(num_features))

    return data, labels


