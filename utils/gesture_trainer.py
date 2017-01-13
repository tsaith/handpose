
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.cross_validation import train_test_split

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

import cPickle as pickle

import handpose as hp
from handpose.preprocessing import detect_na, scale, window_method
from handpose.utils import load_class_data, plot_confusion_matrix 

# Random Seed
rand_seed = 7
np.random.seed(rand_seed)  # for reproducibility

# Number of classes
class_names = ['relax', 'fist'] # Class Names
num_classes = len(class_names)

# Directory paths
train_path = "../data/train"
test_path = "../data/test"

# Output files of model and scaler
model_file = 'mlp_model.hdf'
scaler_file = "scaler.dat"

# Number of data columns
num_cols = 28*2 # W

# Load the training data
train_data, train_labels = load_class_data(class_names, train_path, num_cols=num_cols, 
                                           equal_weight=True, verbose=0)
train_rows, train_cols = train_data.shape

# Load the testing data
test_data, test_labels = load_class_data(class_names, test_path, num_cols=num_cols, 
                                         equal_weight=True, verbose=0)
test_rows, test_cols = test_data.shape


print("Training data rows: {}, cols: {}".format(train_rows, train_cols))
print("Testing data rows: {}, cols: {}".format(test_rows, test_cols))


# Number of data,
num_data, num_features = train_data.shape


# Normalized to the mean of features
#train_data = normalize_to_mean(train_data)
#test_data = normalize_to_mean(test_data)

# Prepare the validation and test samples
X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_labels, 
                                                                                                            test_size=0.0,
                                                      random_state=rand_seed)

X_valid, X_test, y_valid, y_test = train_test_split(test_data, test_labels, test_size=0.5,
                                                    random_state=rand_seed)

# Check if there is NAN in the training dataset
has_na = detect_na(train_data)
if has_na:
   print("--------")
   print("Missing values exist; please check the dataset.")

# Standarize the train and test datasets
## Removing the mean and scaling to unit variance.
scaler = StandardScaler().fit(X_train)


X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, num_classes)
Y_valid = np_utils.to_categorical(y_valid, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

num_train = len(X_train)
num_valid = len(X_valid)
num_test = len(X_test)


print("num_classes = {}".format(num_classes))
print("--------")
print("num_data = {}".format(num_data))
print("num_features = {}".format(num_features))
print("--------")
print("num_train = {}".format(num_train))
print("num_valid = {}".format(num_valid))
print("num_test = {}".format(num_test))


# Hyperparameters
num_epoch = 10
batch_size = 128

learning_rate = 0.1
decay_rate = learning_rate / num_epoch
optimizer = SGD(lr=learning_rate, decay=decay_rate, momentum=0.9, nesterov=True)

print("num_epoch = {}".format(num_epoch))
print("batch_size = {}".format(batch_size))
print("learning rate = {}, delay_rate = {}".format(learning_rate, decay_rate))

# Model configuration

model = Sequential()
model.add(Dense(512, activation='relu', input_dim=num_features))
model.add(Dropout(0.2))

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=num_epoch,
                 verbose=1, validation_data=(X_valid, Y_valid))

# Validation Loss and accurary
score = model.evaluate(X_valid, Y_valid, verbose=0)
print("Validation loss: {}".format(score[0]))
print("Validation accuracy: {}".format(score[1]))

# Testing Loss and accurary
score = model.evaluate(X_test, Y_test, verbose=0)
print("Test loss: {}".format(score[0]))
print("Test accuracy: {}".format(score[1]))

# Save the model as file
model.save(model_file)
print("Save the model as file: {}".format(model_file))

pickle.dump(scaler, open(scaler_file, "wb")) # Dump the scaler
print('Dump the scaler file as {}'.format(scaler_file))

