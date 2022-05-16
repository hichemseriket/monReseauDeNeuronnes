# code propre sans mention de plantes pour la reutilisation comme usine a model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *


def initialisation(X):
    W = np.random.rand(X.shape[1], 1)
    b = np.random.rand(1)
    return W, b


def model(X, w, b):
    Z = X.dot(w) + b
    print("z max", Z.min())
    A = 1 / (1 + np.exp(-Z))
    return A


def log_loss(A, y):
    # si je laisse comme cela :     # return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A)) ,
    # ca fonctionnerait que pour les petites valeurs de x, mais si jamais mes données sont nombreuses, mon Z et A seront
    # impossible à calculer dû à l'exponentiel énorme ou log de zero impossible, car log nest jamais défini sur 0
    # je rajoute du coup une tolerance epselon
    epselon = 1e-15
    return 1 / len(y) * np.sum(-y * np.log(A + epselon) - (1 - y) * np.log(1 - A + epselon))


def gradients(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, (A - y))
    db = 1 / len(y) * np.sum(A - y)
    return dW, db


def update(w, b, dw, db, lr):
    w = w - lr * dw
    b = b - lr * db
    return w, b


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
        # print("epoch:", i, "loss:", log_loss(A, y), "accuracy:", accuracy_score(y, y_pred))
    # print("W:", w)
    # print("b:", b)

    plt.figure("LOSS FUNCTION 1")
    plt.plot(Loss)
    return w, b

# pour visualiser l'importance de la normalisation
def artificial_neural_network2(X, y, lr=0.1, n_iter=1000):
    # initialisation w, b
    w, b = initialisation(X)
    w[0], w[1] = -7.5, -7.5
    nb = 10
    j = 0
    history = np.zeros((n_iter // nb, 5))
    Loss = []
    for i in range(n_iter):
        A = model(X, w, b)
        Loss.append(log_loss(y, A))
        dW, db = gradients(A, X, y)
        w, b = update(w, b, dW, db, lr)
        # enregistrer toutes les 10 iteration, les parametres
        if i % nb == 0:
            history[j, 0] = w[0]
            history[j, 1] = w[1]
            history[j, 2] = b
            history[j, 3] = i
            history[j, 4] = log_loss(y, A)
            j += 1
    plt.figure("LOSS FUNCTION 2")
    plt.plot(Loss)
    return history, b


def predict(X, w, b):
    A = model(X, w, b)
    # if A >= 0.5:
    #     print("OUI")
    # else:
    #     print("NON")
    print("################    La Probabilité que ce soit VRAI est de : ", A, "    ###############")
    return A >= 0.5
# def A_to_Word(A):
#     if A >= 0.5:
#         return "OUI"
#     else:
#         return "NON"

plt.show()
