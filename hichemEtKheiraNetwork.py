from ModelLab import *
# from testD import *
from utilities import *

X_train, y_train, X_test, y_test = load_data()

X_train.flatten()

X_train_reshape = X_train.reshape(X_train.shape[0], -1) / X_train.max()
print(X_train_reshape.max())

X_test_reshape = X_test.reshape(X_test.shape[0], -1) / X_train.max()

# X_train_reshape = X_train.reshape(X_train.shape[0], -1)
# print(X_train_reshape.max())
#
# X_test_reshape = X_test.reshape(X_test.shape[0], -1)

# history, b = artificial_neural_network2(X_train_reshape, y_train)
W, b = artificial_neural_network(X_train_reshape, y_train, lr=0.01, epochs=10000)































































#
# import numpy as np
# from ModelLab import *
# from utilities import *
# import h5py as h5
# import matplotlib.pyplot as plt
#
#
# X_train, y_train, X_test, y_test = load_data()
# print(X_train.shape)
# print(X_test.shape)
# # print(x_train[0])
# # x_train = x_train.reshape((x_train.shape[0]),1)
# # print("apres", x_train.shape)
#
# X_train.flatten()
# # print("apres flatten", x_train.shape)
# print("le max pixel in this array for train data is before devide: ", X_train.max())
#
# X_train_reshape = X_train.reshape(X_train.shape[0], -1) / X_train.max()
# print("le max quon obtient apres normalisation du coup (devide by x_train.max()) : ", X_train_reshape.max())
# print("multiply 64*64", X_train_reshape.shape)
#
#
# X_test_reshape = X_test.reshape(X_test.shape[0], -1) / X_train.max()
# print("le max pixel in this array for test data is after devided  : ", X_test_reshape.max())
#
# print("multiply xtest 64*64", X_test_reshape.shape)
#
# # history, b = artificial_neural_network2(X_train_reshape, y_train)
# w, b = artificial_neural_network(X_train_reshape, y_train)
# # plt.show()
#
# # Normaliser
# # flatten()