# cree mon data set d'abord dune taille de 100 ligne et deux collone represenattant des plante avec la largeur et longeur de ses feuille
#le but est d'entrainer un neuronne artificielle a reconnaitre les plantes tocique des plantes non toxique garce a ces donnée de references
# le but et d'entrainer un modele de regression lineaire sur ces donnees

# 1. Importer les librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2. Lire les donnees
dataset = pd.read_csv('Classeur1.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# 3. Splitter les donnees en training set et test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# 4. Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# 5. Faire la regression lineaire
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# 6. Predire les resultats
y_pred = regressor.predict(X_test)

# 7. Visualiser les resultats
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Taux de feuille (Training set)')
plt.xlabel('Largeur')
plt.ylabel('Taux de feuille')
plt.show()

#
