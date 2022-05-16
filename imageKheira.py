# from sklearn.datasets import make_blobs, make_circles
# import matplotlib.pyplot as plt
from utilities import *
#
from NeuralNetwork import *
# #
# # X, y = make_circles(n_samples=100, noise=0.1, factor=0.3,random_state=0)
# # X = X.T
# # y = y.reshape(1, y.shape[0])
# # print("X.shape = ", X.shape)
# # print("y.shape = ", y.shape)
# #
# # plt.scatter(X[0, :], X[1, :], c=y, cmap="summer")
# #
# # parametres = neural_network(X, y, 2)
# X_train, y_train, X_test, y_test = load_data()
#
# print("taille de mon set d'entrainement sur x", X_train.shape)
# print("taille de mon set d'entrainement sur y",y_train.shape)
# # print("first photo",x_train[0].shape)
# # print(x_test.shape)
# # print(y_test.shape)
# print("nombre d'etiquette qu'il y'a : ", np.unique(y_train, return_counts=True))
#
# print("taille de mon set de test sur x", X_test.shape)
# print("taille de mon set de test sur Y", y_test.shape)
# print("nombre d'etiquette qu'il y'a : ", np.unique(y_test, return_counts=True))
#
# X_train = X_train.T
# X_train[:, 1] = X_train[:, 1] * 20
# y_train = y_train.reshape((1, y_train.shape[0]))
#
# print('dimensions de X:', X_train.shape)
# print('dimensions de y:', y_train.shape)
#
#
# parametres = neural_network(X_train, y_train, 2, lr=.1, n_iter=100)


from utilities import *
from FabriqueModel import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

# TO DO 1. normaliser le train_set et le test_set (0-255 -> 0-1) LA NORMALISATION DES DONNEES EST UNE DES PLUS
# IMPORTANTE TECHNIQUE DE PREVENTION DE LA PERTE DE DONNEES IL FAUT TOUJOURS NORMALISER NOS DONN2E DE MOMENT OU ON
# UTILISE LA DESCENTE DE GRADIENT METTRE SUR UNE MEME ECHELLE TOUTES LES VARIABLE DE NOTRE DATA SET POUR QUE LES
# GRANDES VALEURS NE VIENNENT PAS ECRASER LES PETITES

# En principe : Quand les variables sont sur une meme echelle, la fonction coût évolue de façon similaire sur tous
# ses paramètres. Cela permet une bonne convergence de l'algorithme de la descente de gradient. Mais si une variable
# est plus importante que l'autre, Alors la fonction coût est comprésée, car le poids important de la variable 2 a
# un grand impact sur la sortie A(Z). Cela complique la convergence de la descente de gradients.
x_train, y_train, x_test, y_test = load_data()

print("taille de mon set d'entrainement sur x", x_train.shape)
print("taille de mon set d'entrainement sur y", y_train.shape)

print("nombre d'etiquette qu'il y'a : ", np.unique(y_train, return_counts=True))

print("taille de mon set de test sur x", x_test.shape)
print("taille de mon set de test sur Y", y_test.shape)
print("nombre d'etiquette qu'il y'a : ", np.unique(y_test, return_counts=True))

# ceci ne fonction pas car je nai plus x = mes photos et y egale mes labels , ici jai x mes photo mais jai y et z qui
# sont la moitier des valeur 'pixels) w, b = artificial_neural_network(x_train, y_train)

# j'applati les donné pour leur données 2 dimensions, ma methode a moi qui ne fonctionne pas
# x = x_train.flatten()
# y = y_train.flatten()
# w, b = artificial_neural_network(x, y)

# la methode qui fonctionne j'aurais pu ecrire x_train_reshape = x_train.reshape(x_train.shape[0], x_train.shape[
# 1]*x_train.shape[2]) pour avoir une matrice de taille (n, m) au lieu de (n, k, p) avec m = k*p mais python nous
# permet de faire plus vite en ecrivant -1 dans la deuxieme dimension cela signifie en python : reshape on va
# redimensionner notre tableau x_train pour que celui-ci soit de dimension 1000, virgule tous ce qui reste a
# reorganiser ctd 64 x 64
x_train_reshape = x_train.reshape(x_train.shape[0], -1)

parametres = neural_network(x_train_reshape, y_train, 2, lr=.1, n_iter=100)