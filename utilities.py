import h5py as h5
import numpy as np
# print("Imported h5py and numpy", h5.__version__, np.__version__)

def load_data():
    train_dataset = h5.File("trainset.hdf5", "r")
    X_train = np.array(train_dataset["X_train"][:])  # your train set features
    y_train = np.array(train_dataset["Y_train"][:])  # your train set labels

    test_dataset = h5.File("testset.hdf5", "r")
    X_test = np.array(test_dataset["X_test"][:])  # your train set features
    y_test = np.array(test_dataset["Y_test"][:])  # your train set labels

    return X_train, y_train, X_test, y_test