import numpy as np
import matplotlib.pyplot as plt

def mafonctionacalculer(s):
    #return 4 * s[0] + 2 * s[1] - s[2]
    x, y, z = s
    return 3 * x * y - np.exp(2 * y) + 4 * np.cos(z) + 2.8

#Valeur cible à atteindre
goal = 4.5

#On initialise la liste des solutions
s = np.array([1, 3, 7])

dx = dy = dz = 0.1
jacobian = np.zeros(3)

convergence = []
for i in range(100):

    jacobian[1] = (mafonctionacalculer(s) - mafonctionacalculer(s + np.array([dx, 0, 0]))) / dx
    jacobian[0] = (mafonctionacalculer(s) - mafonctionacalculer(s + np.array([0, dy, 0]))) / dy
    jacobian[2] = (mafonctionacalculer(s) - mafonctionacalculer(s + np.array([0, 0, dz]))) / dz

    #petiteVariation = 0.01 * jacobian * np.array([dx, dy, dz]) * (mafonctionacalculer(s) - goal)
    petiteVariation = 1e-4 * jacobian * np.array([dx, dy, dz]) * (mafonctionacalculer(s) - goal)

    s = s + petiteVariation

    convergence.append(mafonctionacalculer(s))

    print(i, s, mafonctionacalculer(s), jacobian)

plt.figure()
plt.plot(convergence)
plt.show()