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

y = y.reshape((y.shape[0]), 1)

# a decommenter pour faire fonctionner le code normalenet
w, b = artificial_neural_network(X, y)


plante_test_1 = np.array([[2, 1]])
plante_test_2 = np.array([[-1, 1]])
plante_test_3 = np.array([[-1, -1]])

x0 = np.linspace(-1, 4, 100)
x1 = (-w[0] * x0 - b[0]) / w[1]

plt.figure("DATA & LINEAIRE")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer')
plt.xlabel('longeur des feuille')
plt.ylabel('largeur des feuille')
plt.scatter(plante_test_1[:, 0], plante_test_1[:, 1], color='red')
plt.scatter(plante_test_2[:, 0], plante_test_2[:, 1], color='blue')
plt.scatter(plante_test_3[:, 0], plante_test_3[:, 1], color='cyan')

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer')
print("TEST de mon model, avec 3 plante que je mets directement dans le code:")
plt.plot(x0, x1, color='black', lw=3)


b=0
z=X.dot(w)+b


def A_to_Word(plante):
    if predict(plante, w, b):
        return "OUI"
    else:
        return "NON"
print("prediction : est ce que la plante test 1 est toxique ? au vu de la probabilité qui precede :",
      A_to_Word(plante_test_1))
print("prediction : est ce que la plante test 2 est toxique ? au vu de la probabilité qui precede :",
      A_to_Word(plante_test_2))
print("prediction : est ce que la plante test 3 est toxique ? au vu de la probabilité qui precede :",
      A_to_Word(plante_test_3))

# essayer dimplementer la bar progressive :
# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
# plt.figure("LES COÛTS", figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.contourf(w11, w22, L, 20, cmap='magma')
# mes ajout pour rendre l'application interactive
# a decommenter pour faire fonctionner le code normalenet
print("################################################################################################")
print("################################################################################################")
print("#######################                PARTIE INTERACTIVE              #########################")

q = int(input("##### Au clavier !!Tapez 1 pour predire une nouvelle plante sinon 2 pour voir le model #########"))
if q == 1:
    v = print("##############          Vous avez choisi de predire une nouvelle plante        #################")
elif q == 2:
    v = print("##############                 Vous avez choisi de voir le model               #################")

while q == 1:
    print("################     Quels sont les coordonnées de la nouvelle plante ?      ###################")
    x1 = int(input("#################       veuillez renseigner la longeur de sa feuille :      ####################"))
    y1 = int(input("############     Maintenant, veuillez indiquer la largeur de sa feuille :      #################"))

    # A List of Items
    items = list(range(0, 10))
    l = len(items)

    # Initial call to print 0% progress
    printProgressBar(0, l, prefix='Prédiction en cours:', suffix='Prédiction fini', length=50)
    for i, item in enumerate(items):
        # Do stuff...
        time.sleep(0.1)
        # Update Progress Bar
        printProgressBar(i + 1, l, prefix='Prédiction en cours:', suffix='Prédiction fini', length=50)
    new_plant = np.array([[x1, y1]])
    plt.scatter(new_plant[:, 0], new_plant[:, 1], color='red')
    g = predict(new_plant, w, b)
    if g >= 0.5:
        p = print("#######################         Cette plante est bien toxique !        #########################")
    else: p = print("#######################         Cette plante n'est pas toxique !      ##########################")
    print("################################################################################################")
    print("################################################################################################")
    q = int(input("###########   Tapez 1 pour prédire une nouvelle plante sinon 2 pour voir le model   ############"))
    if q == 1:
        v = print("##############          Vous avez choisi de predire une nouvelle plante        #################")
    elif q == 2:
        v = print("##############                 Vous avez choisi de voir le model               #################")
        print("###################################  Au-Revoir  ################################################")

plt.show()
