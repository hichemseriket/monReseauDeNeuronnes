import numpy as np

from utilities import *
from sklearn import *
from ModelLab import *
import matplotlib.pyplot as plt
import plotly.graph_objects as go

X, y = datasets.make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
X[:, 1] = X[:, 1] * 1
y = y.reshape((y.shape[0]), 1)

# plt.figure("DATA & LINEAIRE")
# plt.scatter(X[:,0],X[:,1], c=y, cmap="summer")
# plt.xlabel("X1")
# plt.ylabel("X2")
# plt.title("DATA")

lim = 10
h = 100
w1 = np.linspace(-lim, lim, h)
# print(w1)
w2 = np.linspace(-lim, lim, h)

w11, w22 = np.meshgrid(w1, w2)

w_final = np.c_[w11.ravel(), w22.ravel()].T
# print("w final",w_final.shape)


b = 0
z = X.dot(w_final) + b
A = 1 / (1 + np.exp(-z))

print("taille de A : ", A.shape)

epsilon = 1e-15
L = 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon), axis=0).reshape(w11.shape)
print("L :", L.shape)
# L.shape
plt.figure("Loss avec epsilon")
plt.contourf(w11, w22, L, 20, cmap="magma")
plt.colorbar()


# deuxieme fonction neuronal avec history a la place de w en retour ceci me permet de voir ljistorique de certain de mes ppoint que je decide de voir : w1, w2, i, loss
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
    plt.xlabel("iteration")
    plt.ylabel("Perte")
    plt.legend("plus les perte sont petites, plus le model apprend")
    plt.plot(Loss)
    return history, b


history, b = artificial_neural_network2(X, y)

#
plt.figure("LES COÛTS", figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.contourf(w11, w22, L, 20, cmap='magma')
plt.xlabel('w1')
plt.ylabel('w2')
plt.colorbar()
plt.title('lespace de la fonction Loss sans history')
plt.subplot(1, 2, 2)
plt.contourf(w11, w22, L, 20, cmap='magma')
plt.scatter(history[:, 0], history[:, 1], c=history[:, 2], cmap='Set3_r', marker='x')
# give name to all history points
plt.title('history')

plt.xlabel('w1')
plt.ylabel('w2')
plt.colorbar()
#
#
# # visualiser 3D
# fig = go.Figure(data=go.Surface(z=L, x=w11, y=w22, opacity=1))
# fig.update_layout(title='Surface plot', template='plotly_dark', margin=dict(r=0, l=0, b=0, t=0))
# fig.layout.scene.camera.projection.type = 'orthographic'
# fig.show()

# # visualiser 3D mon test avec w1, w2 a la place de w11, w22
# fig = go.Figure(data=go.Surface(z=L, x=w1, y=w2, opacity=1))
# fig.update_layout(title='Surface plot', template='plotly_dark', margin=dict(r=0, l=0, b=0, t=0))
# fig.layout.scene.camera.projection.type = 'orthographic'
# fig.show()

# plt.show()
