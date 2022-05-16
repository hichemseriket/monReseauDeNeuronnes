import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from newTestIpynb import *
import plotly.graph_objects as go

X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))

print('dimensions de X:', X.shape)
print('dimensions de y:', y.shape)
# plt.figure("Data")
# plt.title('Dataset')
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer')

W, b = artificial_neuron(X, y)

fig, ax = plt.subplots(figsize=(9, 6))
plt.title("data lineaire")
ax.scatter(X[:,0], X[:, 1], c=y, cmap='summer')

x1 = np.linspace(-1, 4, 100)
x2 = ( - W[0] * x1 - b) / W[1]

ax.plot(x1, x2, c='orange', lw=3)

fig2 = go.Figure(data=[go.Scatter3d(
    x=X[:, 0].flatten(),
    y=X[:, 1].flatten(),
    z=y.flatten(),
    mode='markers',
    marker=dict(
        size=5,
        color=y.flatten(),
        colorscale='YlGn',
        opacity=0.8,
        reversescale=True
    )
)])

fig2.update_layout(template= "plotly_dark", margin=dict(l=0, r=0, b=0, t=0))
fig2.layout.scene.camera.projection.type = "orthographic"
fig2.show()