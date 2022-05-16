# code propre sans mention de plantes pour la reutilisation comme usine a model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *
from tqdm import tqdm
import pickle


def initialisation(n0, n1, n2):
    W1 = np.random.rand(n1, n0)
    b1 = np.zeros((n1, 1))
    W2 = np.random.rand(n2, n1)
    b2 = np.zeros((n2, 1))
    parametres = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }
    return parametres

# def log_loss(A, y):
#     epsilon = 1e-15
#     return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))

def forward_propagation(X, parametres):
    W1 = parametres['W1']
    b1 = parametres['b1']
    W2 = parametres['W2']
    b2 = parametres['b2']
    Z1 = W1.dot(X) + b1
    A1 = 1 / (1 + np.exp(-Z1))
    Z2 = W2.dot(A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))
    activations = {
        'A1': A1,
        'A2': A2
    }
    return activations


def back_propagation(X, y, parametres, activations):
    A1 = activations['A1']
    A2 = activations['A2']
    W2 = parametres['W2']

    m = y.shape[1]

    dZ2 = A2 - y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2,
    }
    return gradients


def update(gradients, parametres, lr):
    W1 = parametres['W1']
    b1 = parametres['b1']
    W2 = parametres['W2']
    b2 = parametres['b2']

    dW1 = gradients['dW1']
    db1 = gradients['db1']
    dW2 = gradients['dW2']
    db2 = gradients['db2']

    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2

    parametres = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }
    return parametres

def predict(X, parametres):
    activations = forward_propagation(X, parametres)
    # print("################    La Probabilité que ce soit VRAI est de : ", A, "    ###############")
    A2 = activations['A2']

    return A2 >= 0.5

#
# def neural_network(X_train, y_train, n1, lr=0.1, n_iter=100):
#     # initialisation W, b
#     n0= X_train.shape[0]
#     n2= y_train.shape[0]
#     parametres = initialisation(n0, n1, n2)
#
#     train_loss = []
#     train_acc = []
#     # Boucle d'apprentissage
#     for i in range(n_iter):
#         activations = forward_propagation(X_train, parametres)
#         gradients = back_propagation(X_train, y_train, activations, parametres)
#         parametres = update(parametres, gradients, lr)
#         # enregistrer toutes les 10 iteration, les parametres
#         if i %10 == 0:
#             train_loss.append(log_loss(y_train, activations['A2']))
#             y_predict = predict(X_train, parametres)
#             current_acc = accuracy_score(y_train.flatten(), y_predict.flatten())
#             train_acc.append(current_acc)
#
#     plt.figure("mon reseau de neuronnes deux chouches", figsize=(14, 4))
#     plt.subplot(1, 2, 1)
#     plt.plot(train_loss, label='train loss')
#     plt.legend()
#     # plt.title("loss")
#     plt.subplot(1, 2, 2)
#     plt.plot(train_acc, label='train acc')
#     plt.legend()
#     # plt.title("accuracy")
#     plt.show()
#     return parametres


def neural_network(X_train, y_train, n1, lr=0.1, n_iter=100):
    # initialisation parametres
    n0 = X_train.shape[0]
    n2 = y_train.shape[0]
    np.random.seed(0)
    parametres = initialisation(n0, n1, n2)

    train_loss = []
    train_acc = []
    history = []

    # gradient descent
    for i in tqdm(range(n_iter)):
        activations = forward_propagation(X_train, parametres)
        A1 = activations['A1']
        A2 = activations['A2']
        W2 = parametres['W2']

        # Plot courbe d'apprentissage
        train_loss.append(log_loss(y_train.flatten(), A2.flatten()))
        y_pred = predict(X_train, parametres)
        train_acc.append(accuracy_score(y_train.flatten(), y_pred.flatten()))

        history.append([parametres.copy(), train_loss, train_acc, i])

        # mise a jour
        gradients = back_propagation(X_train, y_train, parametres, activations)
        parametres = update(gradients, parametres, lr)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='train acc')
    plt.legend()
    plt.show()
    save_model(parametres)
    return parametres

# save the model
def save_model(parametres):
    with open('model.pkl', 'wb') as f:
        pickle.dump(parametres, f)




# plt.show()
