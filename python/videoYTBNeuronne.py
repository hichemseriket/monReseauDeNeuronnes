# cree mon data set d'abord dune taille de 100 ligne et deux collone represenattant des plante avec la largeur et longeur de ses feuille
# le but est d'entrainer un neuronne artificielle a reconnaitre les plantes tocique des plantes non toxique garce a ces donnée de references
# le but et d'entrainer un modele de regression lineaire sur ces donnees

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import sklearn.datasets as datasets
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

X, y = datasets.make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
# X, y = sklearn.datasets.make_blobs(n_samples=100, n_features=2, centers=2,  random_state=0)
# X, y = datasets.make_regression(n_samples=100, n_features=2, n_informative=2, random_state=0)
y = y.reshape((y.shape[0]), 1)
# print("dimension de x:", X.shape)
# print("dimension de y:", y.shape)


def initialisation(X):
    W = np.random.rand(X.shape[1], 1)
    b = np.random.rand(1)
    return (W, b)
# w, b = initialisation(X)
# print("taille de W", w.shape)
# print("taille de b", b.shape)

def model(X, w, b):
    Z = X.dot(w) + b
    A = 1 / (1 + np.exp(-Z))
    return A

# A = model(X, w, b)
# print("taille de A", A.shape)

def log_loss(A, y):
    return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))

# print(log_loss(A, y))

def gradients(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, (A - y))
    db = 1 / len(y) * np.sum(A - y)
    return (dW, db)
# dw, db = gradients(A, X, y)
# print("taille de w",dw.shape)
# print("taille de db ",db.shape)

def update(w, b, dw, db, lr):
    w = w - lr * dw
    b = b - lr * db
    return (w, b)

# update(w, b, dw, db, 0.01)

# print("taille de W updated", w.shape)
# print("taille de b updated", b.shape)

def artificial_neural_network(X, y, lr=0.1, epochs=100):
    # initialisation w, b
    w, b = initialisation(X)
    Loss = []
    history = []
    for i in range(epochs):
        A = model(X, w, b)
        Loss.append(log_loss(A, y))
        dW, db = gradients(A, X, y)
        w, b = update(w, b, dW, db, lr)
        y_pred = predict(X, w, b)
        history.append(y_pred)
        # avec la fonction accuracy_score on peut verifier la precision du modele, on comparant les données de reference y avec nos prediction y_pred
        print("epoch:", i, "loss:", log_loss(A, y), "accuracy:", accuracy_score(y, y_pred))
        # print("W:", w)
        # print("b:", b)
    print("W:", w)
    print("b:", b)
    count_False = 0
    count_True = 0
    for y_pred in y:
        if y_pred == 0:
            count_False += 1
        else:
            count_True += 1
        # if p == "False":
        #     count_False += 1
        # elif p == "True":
        #     count_True += 1
    print("count_False:", count_False)
    print("count_True:", count_True)
    plt.figure("LOSS")
    plt.plot(Loss)
    # plt.show()
    # return (w, b, Loss)
    return (w, b)

# maintenant que j'ai mon model je vais men servir pour effectuer des predictions
# par exemple mettre une nouvelle plante que je mesure sa longeur et sa largeur de ses feuille et je rentre ses info dans mon model,
# mon model va me renvoyé la probabilité que la plante soit toxique ou non,
# car la sortie de mon model c'est une fonction sigmoid que lon peut voir comme une proba compris entre zero et 1
# donc ce quon fait en général c'est qu'a partir du moment ou cette probabilité est superieur à 0.5 ma plante est toxique elle appartient a la classe Y1
# sinon elle est non toxique
# je doit cree la fonction predict dans laquel je passe mes données X ainsi que les parametres w et b de mon model
def predict(X, w, b):
    A = model(X, w, b)
    # A est l'activation associe a cette plante
    print("La probabilité que la plante suivante soit toxique est de :", A)
    return A >= 0.5
    # return Loss


# def predict(X, w, b):
#     Z = X.dot(w) + b
#     A = 1 / (1 + np.exp(-Z))
#     return A

# plt.plot(artificial_neural_network().Loss)
# plt.show()
# print("W:", artificial_neural_network(X, y)[0])
# print("b:", artificial_neural_network(X, y)[1])

# artificial_neural_network(X, y, lr=0.1, epochs=100)
w, b = artificial_neural_network(X, y)
# print("W:", w)
# print("b:", b)
#
# count_False = 0
# count_True = 0
# print("prediction avant plante new:", predict(X, w, b))
#
#
# for p in predict(X, w, b):
#     if p=="False":
#         count_False+=1
#     elif p=="True":
#         count_True+=1
# print("count_False:", count_False)
# print("count_True:", count_True)

new_plant = np.array([[2, 1]])
new_plant2 = np.array([[-1, 1]])
new_plant3 = np.array([[-1, -1]])

x0 = np.linspace(-1, 4, 100)
x1= (-w[0]*x0 - b[0])/w[1]

########################################################################################################################
#copilot how to put ligne with color
# print when Z = 0
# print("Z = 0:", model(new_plant, w, b))
# # plot decision boundary
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
# Z = model(np.c_[xx.ravel(), yy.ravel()], w, b)
# Z = Z.reshape(xx.shape)
# plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
#
# plt.scatter(new_plant[:, 0], new_plant[:, 1], color='red')
#
# plt.show()

########################################################################################################################

plt.figure("DATA & LINEAIRE")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer')
plt.xlabel('longeur des feuille')
plt.ylabel('largeur des feuille')
plt.scatter(new_plant[:, 0], new_plant[:, 1], color='red')
plt.scatter(new_plant2[:, 0], new_plant2[:, 1], color='blue')
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer')
plt.plot(x0, x1, color='black', lw=3)
print("prediction : est ce que la plante 1 est toxique ? au vu de la probabilité qui precede :", predict(new_plant, w, b))
print("prediction : est ce que la plante 2 est toxique ? au vu de la probabilité qui precede :", predict(new_plant2, w, b))
print("prediction : est ce que la plante 3 est toxique ? au vu de la probabilité qui precede :", predict(new_plant3, w, b))

#
# # plot 3D Figure with plotly.graph_objs.Scatter3d
# import plotly.graph_objs as go
# fig = go.Figure(data=[go.Scatter3d(
#     x=X[:, 0].flatten(),
#     y=X[:, 1].flatten(),
#     z=y.flatten(),
#     mode='markers',
#     marker=dict(
#         size=12,
#         color=y.flatten(),                # set color to an array/list of desired values
#         colorscale='YlGn',
#         opacity=0.8,
#         reversescale=True
#     )
# )])
# fig.update_layout(template='plotly_dark', margin=dict(l=0, r=0, b=0, t=0))
# fig.layout.scene.camera.projection.type = 'perspective'
# fig.show()
# # plot 3D Figure of sigmoid function
# x0 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
# x1 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
# xx0, xx1 = np.meshgrid(x0, x1)
# Z = model(np.c_[xx0.ravel(), xx1.ravel()], w, b)
# Z = Z.reshape(xx0.shape)
# fig = go.Figure(data=go.Surface(z=Z,
#                                 x=xx0,
#                                 y=xx1,
#                                 colorscale='YlGn',
#                                 opacity=0.8))
# fig.update_layout(template='plotly_dark', margin=dict(l=0, r=0, b=0, t=0))
# fig.layout.scene.camera.projection.type = 'perspective'
# fig.show()
#

###############################################################################################

print("#############################################################################################")
print("#############################################################################################")
print("#######################                PARTIE INTERACTIVE              ######################")
hichem = int(input("entrez 1 si vous voulez faire tourné le programe et predire plusieur plantes sinon 2 pour une seule fois: "))
while hichem == 1:
    print("#######       A votre clavier !! quels sont les coordonnées de la nouvelle plante ?      #########")
    x1 = int(input("entrer la longeur de la feuille de la nouvelle plante: "))
    y1 = int(input("entrer la largeur de la feuille de la nouvelle plante: "))
    new_plant = np.array([[x1, y1]])
    # new_A = predict(new_plant, w, b)
    # print("quelle est la probabilité que la nouvelle plante soit toxique:", new_A)
    print("est ce que la nouvelle plante est toxique au vu de sa probabilité qui precede :", predict(new_plant, w, b))
    plt.scatter(new_plant[:, 0], new_plant[:, 1], color='cyan')

    hichem = int(input("entrez 1 si vous voulez faire une nouvelle fois des predicitions,"
                       " sinon tapez 2 pour arreter le programme et voir l'affichage: "))
# if hichem == 2:
#     plt.show()
plt.show()
