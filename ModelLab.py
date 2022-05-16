# code propre sans mention de plantes pour la reutilisation comme usine a model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *
from tqdm import tqdm


def initialisation(X):
    W = np.random.rand(X.shape[1], 1)
    b = np.random.rand(1)
    return W, b

def model(X, W, b):
    Z = X.dot(W) + b
    print("Z min ", Z.min())
    A = 1 / (1 + np.exp(-Z))
    return A

def log_loss(A, y):
    epsilon=1e-15
    return 1 / len(y) * np.sum(-y * np.log(A+epsilon) - (1 - y) * np.log(1 - A+epsilon))


def gradients(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, (A - y))
    db = 1 / len(y) * np.sum(A - y)
    return dW, db


def update(dW, db, W, b, lr):
    W = W - lr * dW
    b = b - lr * db
    return W, b


def artificial_neural_network(X, y, lr=0.1, epochs=100):
    # initialisation W, b
    W, b = initialisation(X)
    Loss = []
    acc=[]

    for i in tqdm(range(epochs)):
        # activation
        A = model(X, W, b)
        #calcul le cout
        Loss.append(log_loss(A, y))
        # clacul de la precision
        y_pred = predict(X, W, b)
        acc.append(accuracy_score(y,y_pred))
        #mise à joour
        dW, db = gradients(A, X, y)
        W, b = update(dW, db, W, b, lr)

    # plt.figure("LOSS FUNCTION 1")
    # plt.plot(Loss)
    plt.figure("des trucs", figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(Loss)
    plt.subplot(1,2,2)
    plt.plot(acc)

    return W, b

def artificial_neural_network2(X, y, lr=0.1, n_iter=100):
    # initialisation W, b
    W, b = initialisation(X)
    W[0], W[1] = -7.5, -7.5
    nb = 10
    j = 0
    history = np.zeros((n_iter // nb, 5))
    Loss = []
    for i in range(n_iter):
        A = model(X, W, b)
        Loss.append(log_loss(y, A))
        dW, db = gradients(A, X, y)
        W, b = update(W, b, dW, db, lr)
        # enregistrer toutes les 10 iteration, les parametres
        if i % nb == 0:
            history[j, 0] = W[0]
            history[j, 1] = W[1]
            history[j, 2] = b
            history[j, 3] = i
            history[j, 4] = log_loss(y, A)
            j += 1
    plt.figure("LOSS FUNCTION 2")
    plt.xlabel("iteration")
    plt.ylabel("Perte")
    plt.legend("plus les perte sont petites, plus le model apprend")
    plt.plot(Loss)
    return history, b

def predict(X, W, b):
    A = model(X, W, b)
    # print("################    La Probabilité que ce soit VRAI est de : ", A, "    ###############")
    return A >= 0.5

# plt.show()
