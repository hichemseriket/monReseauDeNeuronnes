from FabriqueModel import *
# LA NORMALISATION DES DONNEES EST UNE DES PLUS
# IMPORTANTE TECHNIQUE DE PREVENTION DE LA PERTE DE DONNEES IL FAUT TOUJOURS NORMALISER NOS DONN2E DE MOMENT OU ON
# UTILISE LA DESCENTE DE GRADIENT METTRE SUR UNE MEME ECHELLE TOUTES LES VARIABLE DE NOTRE DATA SET POUR QUE LES
# GRANDES VALEURS NE VIENNENT PAS ECRASER LES PETITES

# En principe : Quand les variables sont sur une meme echelle, la fonction coût évolue de façon similaire sur tous
# ses paramètres. Cela permet une bonne convergence de l'algorithme de la descente de gradient. Mais si une variable
# est plus importante que l'autre, Alors la fonction coût est comprésée, car le poids important de la variable 2 a
# un grand impact sur la sortie A(Z). Cela complique la convergence de la descente de gradients.

# experience sur la normalisation des donnees
lim = 10
h=100
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
Z = X.dot(w_final)+b

A = 1/(1+np.exp(-Z))
print(A.shape)

# w doit etre de dimension n, 10000 et n = 2 ctd (2, 10000), x de dimmension (100, 2)


