







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
