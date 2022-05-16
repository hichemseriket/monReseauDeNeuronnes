# cree mon data set d'abord dune taille de 100 ligne et deux collone represenattant des plante avec la largeur et longeur de ses feuille
# le but est d'entrainer un neuronne artificielle a reconnaitre les plantes tocique des plantes non toxique garce a ces donnée de references
# le but et d'entrainer un modele de regression lineaire sur ces donnees
from FabriqueModel import *
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import pandas as pd
import sklearn
import sklearn.datasets as datasets
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import time
X, y = datasets.make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)


# teste sur la normalisation en changeant la deuxieme x2 de mes data en la multipliant par 2, puis voir son impact
X[:, 1] = X[:, 1] * 1

y = y.reshape((y.shape[0]), 1)



history, b = artificial_neural_network2(X, y)


plt.figure("DATA & LINEAIRE")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer')
plt.xlabel('longeur des feuille')
plt.ylabel('largeur des feuille')

########################################################################################################################
# a decommenter pour faire fonctionner le code normalenet
# experience sur la normalisation
lim = 10
h = 100
w1 = np.linspace(-lim, lim, h)
w2 = np.linspace(-lim, lim, h)

# voir video meshgrid : tableau 100 x 100, le vecteur w1 a été recopié en boucle pour couvrir la dimension de w2 et
# la meme pour w2
w11, w22 = np.meshgrid(w1, w2)
print(w11.shape)

w_final = np.c_[w11.ravel(), w22.ravel()].T
print(w_final.shape)
# on voudrait passé cette grille de valeur les 10000 dans notre foction Z
b = 0
Z = X.dot(w_final) + b

A = 1 / (1 + np.exp(-Z))

# log loss
epsolon = 1e-15
L = 1 / len(y) * np.sum(-y * np.log(A + epsolon) - (1 - y) * np.log(1 - A + epsolon), axis=0)
print(L.shape)
# nous avons donc 10000 coût differents, il suffit de les redimensionner : L = (100, 100)
# faire un contour plot de cette grille de valeur
# on fait
L = L.reshape(w11.shape)
# on aurait pu directement reshape dans la fonction :
# L = 1 / len(y) * np.sum(-y * np.log(A + epsolon) - (1 - y) * np.log(1 - A + epsolon), axis=0).reshape(w11.shape)

plt.figure("LES COÛTS", figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.contourf(w11, w22, L, 10, cmap='magma')
plt.xlabel('w1')
plt.ylabel('w2')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.contourf(w11, w22, L, 10, cmap='magma')
plt.scatter(history[:, 0], history[:, 1], c=history[:, 2], cmap='Blues', marker='x')
# give name to all history points
plt.title('history')

plt.xlabel('w1')
plt.ylabel('w2')
plt.colorbar()

# visualiser 3D
fig = go.Figure(data=go.Surface(z=L, x=w11, y=w22, opacity=1))
fig.update_layout(title='Surface plot', template='plotly_dark', margin=dict(r=0, l=0, b=0, t=0))
fig.layout.scene.camera.projection.type = 'orthographic'
fig.show()


plt.show()
