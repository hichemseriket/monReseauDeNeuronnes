from sklearn.datasets import make_blobs, make_circles
import matplotlib.pyplot as plt

from NeuralNetwork import *
#
# X, y = make_circles(n_samples=100, noise=0.1, factor=0.3,random_state=0)
# X = X.T
# y = y.reshape(1, y.shape[0])
# print("X.shape = ", X.shape)
# print("y.shape = ", y.shape)
#
# plt.scatter(X[0, :], X[1, :], c=y, cmap="summer")
#
# parametres = neural_network(X, y, 2)

X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
X = X.T
y = y.reshape((1, y.shape[0]))

print('dimensions de X:', X.shape)
print('dimensions de y:', y.shape)
plt.figure("Données")
plt.scatter(X[0, :], X[1, :], c=y, cmap="summer")

parametres = neural_network(X, y, 32)
