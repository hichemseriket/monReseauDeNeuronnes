import matplotlib.pyplot as plt

from utilities import *

# !pip install h5py
X_train, y_train, X_test, y_test = load_data()

print("taille de mon set d'entrainement sur x", X_train.shape)
print("taille de mon set d'entrainement sur y", y_train.shape)

print("taille de mon set de test sur x", X_test.shape)
print("taille de mon set de test sur Y", y_test.shape)

# y_train = y_train.T
# y_test = y_test.T
#
# X_train = X_train.T
# X_train_reshape = X_train.reshape(-1, X_train.shape[-1]) / X_train.max()
# X_test = X_test.T
# X_test_reshape = X_test.reshape(-1, X_test.shape[-1]) / X_train.max()
#
# print("taille de mon set d'entrainement transposé sur x", X_train.shape)
# print("taille de mon set d'entrainement transposé and reshaped sur x", X_train_reshape.shape)
# print("taille de mon set d'entrainement transposé sur y", y_train.shape)
# #
# m_train = 300
# m_test = 80
# a = X_test.reshape[:, :m_test]
# print("taille de mon set d'entrainement reshaped sur x", a.shape)

# x_train_reshape = X_train.reshape[:, :m_train]
# y_train=y_train[:, :m_train]
# y_test=y_test[:, :m_test]
#
# print("taille de mon set d'entrainement sur x", x_train_reshape.shape)
# print("taille de mon set de test sur x", x_test_reshape.shape)
# print("taille de mon set d'entrainement sur y", y_train.shape)
# print("taille de mon set de test sur y", y_test.shape)
# plt.figure(figsize=(10,10))
# plt.imshow(x_train[0])
# plt.show()


# facon copilot d'afficher les 10 premiere photo
# plt.figure(figsize=(16, 8))
# for i in range(9):
#     plt.subplot(3, 3, i+1)
#     plt.imshow(x_train[i])
#     plt.title(y_train[i])
#     plt.axis('off')
# plt.show()


# facon youtube pour afficher les 10 premiere photo
plt.figure(figsize=(16, 8))
for i in range(1, 10):
    plt.subplot(4, 5, i)
    plt.imshow(X_train[i], cmap='gray')
    if y_train[i] == 0:
        plt.title('cat')
    else:
        plt.title('dog')
    # plt.title(y_train[i])
    plt.tight_layout()
plt.show()
