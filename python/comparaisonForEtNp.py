# import time
# import numpy as np
#
#
# time_start = time.time()
# liste_A=[]
# for i in range (0,1000000) :
#     liste_A.append(i)
#
# liste_B=[]
# for i in range (0, len(liste_A)):
#     liste_B.append(liste_A[i]*2)
# print(liste_A)
# print(liste_B)
# time_end = time.time()
# print("Temps d'exécution en normale :", time_end - time_start)
#
# # time_start1 = time.time()
# # for i in range (0,10000) :
# #     liste_C = np.array([i])
# # print(liste_C)
# # for i in range (0,10000) :
# #     liste_D = 2*liste_C
# #
# # print(liste_D)
# #
# # time_end1 = time.time()
# # print("Temps d'exécution en numpy :", time_end1 - time_start1)
#
#
# time_start2 = time.time()
# liste_C = np.array([i for i in range (0,1000000)])
# liste_D = 2*liste_C
# print(liste_C)
# print(liste_D)
# time_end2 = time.time()
# print("Temps d'exécution en numpy avec la boucle for a linterieur du np.array :", time_end2 - time_start2)
#
# time_start3 = time.time()
# liste_E = np.array(range(0,1000000,1))
# liste_F = 2*liste_E
# print(liste_E)
# print(liste_F)
# time_end3 = time.time()
# print("Temps d'exécution en numpy sans for dans le np.array :", time_end3 - time_start3)
#
# hichem=[]
# for i in range (10,0,-1):
#     hichem.append(i)
#
# print(hichem)
#
# print(hichem[:10:2])

# import matplotlib.pyplot as plt
# import numpy as np
#
# a = np.random.random((16, 16))
# plt.imshow(a, cmap='hot', interpolation='nearest')
# plt.show()


import matplotlib.pyplot as plt
import numpy as np


# # generate 2 2d grids for the x & y bounds
# y, x = np.meshgrid(np.linspace(-3, 3, 1000), np.linspace(-3, 3, 1000))
#
# z = (1 - x / 2. + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
# # x and y are bounds, so z should be the value *inside* those bounds.
# # Therefore, remove the last value from the z array.
# z = z[:-1, :-1]
# z_min, z_max = -np.abs(z).max(), np.abs(z).max()
#
# fig, ax = plt.subplots()
#
# c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
# ax.set_title('coucouColor')
# # set the limits of the plot to the limits of the data
# ax.axis([x.min(), x.max(), y.min(), y.max()])
# fig.colorbar(c, ax=ax)
#
# plt.show()


# c reate class humain
class Humain:
    def __init__(self, nom, vie, degats, armure):
        self.nom = nom
        self.age = ""
        self.fonction = ""
        self.salary = 0
        self.pointsdevie = 100
        self.poids = 0
        self.taille = 0
        self.sexe = ""
        self.degats = degats
        self.vie = vie
        self.vie_max = 200
        self.force = 100
        self.intelligence = 15
        self.agilite = 15
        self.chance = 15
        self.vitesse = 15
        self.armure = armure
        self.armure_degats = 15
        self.armure_degats_max = 30
        self.armure_degats_min = 0
        self.sante = vie + armure

    def attaquer(self, other):
        # other.sante -= self.degats
        # other.sante -= self.degats
        # other.armure -= self.armureDegats
        other.degats(self.age)
        print("je suis un humain et je t'inflige", self.degats, "points de degats")

    def seFaireAttaquer(self, other):
        # Cas ou tes degats recus sont inferieurs a ton armure
        if other.degat < self.armure:
            self.armure -= other.degat
            print("ton armure lui reste", self.armure)
        # Cas ou tes degats recus sont superieurs a ton armure
        else:
            self.vie = other.degat - self.armure
            self.armure = 0

        self.sante -= other.degat
        print(self.nom, "reçois", other.degat, "points de degats, de la part de", other.nom)
        if self.sante <= 0:
            print(self.nom + " est mort")
            return
        else:
            print(" les points de vie de ", self.nom, " passe de ", (self.sante + other.degat), " à ", str(
                self.sante), " points de vie")

    # def seFaireAttaquer(self, other):
    #     # Cas ou tes degats recus sont inferieurs a ton armure
    #     if other.degat < self.armure:
    #         self.armure -= other.degat
    #     # Cas ou tes degats recus sont superieurs a ton armure
    #     else:
    #         newDegat = other.degat - self.armure
    #         newPoint = self.vie - newDegat
    #         self.vie = newPoint
    #         self.armure = 0
    #     if self.vie <= 0:
    #         print(self.nom + " est mort")
    #     else:
    #         print(self.nom + " a " + str(self.vie) + " points de vie")
    #     self.sante -= other.degat
    #     print("je reçois", other.degat, "points de degats")

    def recoit_degats(self, degats):
        self.vie = self.vie - degats
        if self.vie <= 0:
            print(self.nom + " est mort")
        else:
            print(self.nom + " a " + str(self.vie) + " points de vie")

    def recoit_soins(self, soins):
        self.vie = self.vie + soins
        if self.vie > self.vie_max:
            self.vie = self.vie_max
        print(self.nom + " a " + str(self.vie) + " points de vie")

    def __str__(self):
        return "Nom : " + self.nom + "\nDegats : " + str(
            self.degats) + "\nVie : " + str(self.vie) + "\nForce : " + str(self.force) + "\nArmure : " + str(
            self.armure) + "\nArmure de degats : " + str(self.armure_degats) + "\nArmure de degats max : " + str(
            self.armure_degats_max)


class Monstre:
    # def __init__(self, nom, vie, vieMax, degats, armure):
    def __init__(self, nom, vie, degats, armure):
        self.nom = nom
        self.couleur = ""
        self.puissance = 100
        self.pointDeVie = vie
        self.pointDeVieMax = 200
        if self.nom == "hichem":
            self.pointDeVieMax = 300
        if self.nom == "darshan":
            self.pointDeVieMax = 100
            print("toi enfoiré je te bloque à 100 points de vie, hihihihi")
        self.pointDeVieMin = 0
        self.pointDeVieMoyen = (self.pointDeVieMax + self.pointDeVieMin) / 2
        self.pointDeVieMoyen = self.pointDeVieMoyen
        self.degat = degats
        self.degatMax = 10
        self.degatMin = 0
        self.degatMoyen = (self.degatMax + self.degatMin) / 2
        self.degatMoyen = self.degatMoyen
        self.force = 100
        self.forceMax = 10
        self.forceMin = 0
        self.forceMoyen = (self.forceMax + self.forceMin) / 2
        self.forceMoyen = self.forceMoyen
        self.armure = armure
        self.armureMax = 10
        self.armureMin = 0
        self.armureMoyen = (self.armureMax + self.armureMin) / 2
        self.armureMoyen = self.armureMoyen
        self.sante = self.pointDeVie + self.armure
        if self.sante <= 0:
            print(self.nom + " est mort")
            return
        if self.sante > self.pointDeVieMax:
            self.sante = self.pointDeVieMax
            print("sante max atteinte, tu es bloqué a " + str(self.pointDeVieMax) + " points de vie")
        self.armureDegats = 5
        self.armureDegatsMax = 30
        self.armureDegatsMin = 0
        self.armureDegatsMoyen = (self.armureDegatsMax + self.armureDegatsMin) / 2
        self.armureDegatsMoyen = self.armureDegatsMoyen

    def __str__(self):
        return "Nom : " + self.nom + "\nDegats : " + str(
            self.degat) + "\nVie : " + str(self.pointDeVie) + "\nArmure : " + str(
            self.armure) + "\nSanté totale : " + str(self.sante)

    def parler(self):
        print("je suis un monstre")

    def marcher(self):
        print("je marche")

    def courir(self):
        print("je cours")

    def sauter(self):
        print("je saute")

    def attaquer(self, other):
        if self.armureDegats > 0:
            newDegat = self.degat - self.armureDegats
            newPoint = other.sante - newDegat
            other.sante = newPoint
            self.armureDegats = 0
        # other.sante -= self.degat
        # other.armure -= self.armureDegats

    # def frappe(self, ennemi):
    #     print(self.nom + " frappe " + ennemi.nom)
    #     self.pointDeVie -= ennemi.degat(self)
    #     print("je suis un monstre et je t'inflige", self.degat, "points de degats")
    def recoit_degats(self, degats):
        self.pointDeVie = self.pointDeVie - degats
        if self.pointDeVie <= 0:
            print(self.nom + " est mort")
        else:
            print(self.nom + " a " + str(self.pointDeVie) + " points de vie")

    def seFaireAttaquer(self, other):
        # Cas ou tes degats recus sont inferieurs a ton armure
        if other.degat < self.armure:
            self.armure -= other.degat
            print("ton armure lui reste", self.armure)

        # Cas ou tes degats recus sont superieurs a ton armure
        else:
            self.pointDeVie = other.degat - self.armure
            self.armure = 0

        self.sante -= other.degat
        print(self.nom, "reçois", other.degat, "points de degats, de la part de", other.nom)
        if self.sante <= 0:
            print(self.nom + " est mort")
            return
        else:
            print(" les points de vie de ", self.nom, " passe de ", (self.sante + other.degat), " à ", str(
                self.sante), " points de vie")
            # print(" les points de vie de ", self.nom, " passe de ", self.sante , " à ", str(
            #                 self.sante-other.degat), " points de vie")

    def soins_recu(self, soins):
        self.pointDeVie = self.pointDeVie + soins
        if self.pointDeVie > self.pointDeVieMax:
            self.pointDeVie = self.pointDeVieMax
        print(self.nom + " a " + str(self.pointDeVie) + " points de vie")

    def mourir(self):
        if self.pointDeVie <= 0:
            print("je suis mort")
        else:
            print("je suis encore en vie")

    def sePresenter(self):
        print("je suis un monstre et je m'appelle", self.nom)
        print("je suis de couleur", self.couleur)
        print("je possede", self.puissance, "puissance")
        print("je possede", self.pointDeVie, "point de vie")
        print("je possede", self.degat, "degat")

    def getPointDeVie(self):
        return self.pointDeVie

    def seDeplacer(self):

        print("je me deplace")

    def seReproduire(self):
        print("je me reproduis")

    def seNourrir(self):
        print("je me nourris")

    def seSoigner(self):
        print("je me soigne")

    def seFaireManger(self):
        print("je me fais manger")


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# print(" Bonjour dans mon jeu developpé en 1h30 et qui va detronné GTA !! ")
# humain_name= input("veuillez donner le nom du premier humain")
# humain_age = int(input("veuillez mettre un age"))
# humain_vie = int(input("veuillez mettre ses points de vie"))
# humain_degat = int(input("veuillez mettre ses degat"))
# humain_force = int(input("veuillez mettre une force"))
# humain_armure = int(input("veuillez mettre la capacite de larmure"))
# humain_armureDegat = int(input("veuillez mettre un degat de larmure"))
# darshan= Humain(humain_name, humain_age, humain_vie, humain_degat, humain_force, humain_armure, humain_armureDegat)
# print(darshan)
if __name__ == "__main__":
    print("Bonjour dans notre jeu developpé par hichem")

    monstre_name = input("veuillez donner un nom au premier monstre : ")
    if monstre_name == "hichem":
        print(bcolors.OKGREEN + "you are the boss  " + monstre_name + bcolors.ENDC)
    elif monstre_name == "darshan":
        print(bcolors.WARNING + "tu es nul " + monstre_name + " et tu vas mourir" + bcolors.ENDC)
    else:
        print("sort de la, " + monstre_name)
    monstre_vie = int(input("veuillez mettre ses points de vie :"))
    # monstre_vieMax = int(input("veuillez mettre sa vie max :"))
    # if monstre_vie > monstre_vieMax:
    #     monstre_vie = monstre_vieMax
    #     print("votre vie est superieur a votre vie max")
    #     print("votre vie est bloqué a : " + str(monstre_vie))
    monstre_degat = int(input("veuillez mettre ses degat :"))
    if monstre_degat > 20:
        print(bcolors.OKBLUE + "WOW tu es le boss du game" + bcolors.ENDC)
    elif monstre_degat == 20:
        print(bcolors.OKCYAN + "tu es tres fort" + bcolors.ENDC)
    elif monstre_degat < 10:
        print("tu as une force de merde")
    monstre_armure = int(input("veuillez mettre la capacite de larmure :"))
    if monstre_armure > 20:
        print(bcolors.OKBLUE + "ohh lalalala mais c'est une armure legendaire" + bcolors.ENDC)
    elif monstre_armure == 20:
        print(bcolors.OKCYAN + "tu as une armure tres RARE" + bcolors.ENDC)
    elif monstre_armure <= 15:
        print("tu as une armure de merde")
    darshan = Monstre(monstre_name, monstre_vie, monstre_degat, monstre_armure)
    # print("le premier monstre est :  ", darshan)

    monstre_name = input("veuillez donner un nom au deuxieme monstre : ")
    if monstre_name == "hichem":
        print(bcolors.OKGREEN + "you are the boss  " + monstre_name + bcolors.ENDC)
    elif monstre_name == "darshan":
        print(bcolors.WARNING + "tu es nul " + monstre_name + " et tu vas mourir" + bcolors.ENDC)
    else:
        print("sort de la, " + monstre_name)
    monstre_vie = int(input("veuillez mettre ses points de vie :"))
    # monstre_vieMax = int(input("veuillez mettre sa vie max :"))
    # if monstre_vie > monstre_vieMax:
    #     monstre_vie = monstre_vieMax
    monstre_degat = int(input("veuillez mettre ses degat :"))
    if monstre_degat > 20:
        print(bcolors.OKBLUE + "WOW tu es le boss du game" + bcolors.ENDC)
    elif monstre_degat == 20:
        print(bcolors.OKCYAN + "tu es tres fort" + bcolors.ENDC)
    elif monstre_degat <= 15:
        print("tu as une force de merde")
    monstre_armure = int(input("veuillez mettre la capacite de larmure :"))
    if monstre_armure > 20:
        print(bcolors.OKBLUE + "ohh lalalala mais c'est une armure legendaire" + bcolors.ENDC)
    elif monstre_armure == 20:
        print(bcolors.OKCYAN + "tu as une armure tres RARE" + bcolors.ENDC)
    elif monstre_armure < 10:
        print("tu as une armure de merde")
    hichem = Monstre(monstre_name, monstre_vie, monstre_degat, monstre_armure)
    # print(" le deuxieme monstre avant attaque : ", hichem)

    # jean=Humain("jean",200,35,15)
    # jean.seFaireAttaquer(hichem)
    # print(hichem)
    # print(jean)

    while hichem.sante > 0 and darshan.sante > 0:
        # print("le deuxieme monstre attaque : ", hichem)
        # print("le premier monstre avant attaque : ", darshan)
        darshan.seFaireAttaquer(hichem)
        hichem.seFaireAttaquer(darshan)

        # print("le premier monstre attaque : ", darshan)
    if hichem.sante > 0:
        print(bcolors.OKBLUE, hichem.nom, "a gagné")
    else:
        print(bcolors.OKBLUE, darshan.nom, "a gagné")


    # if darshan.sante > 0:
    #     print(bcolors.OKBLUE, darshan.nom, "a gagné")
    # else:
    #     print(bcolors.OKBLUE, hichem.nom, "a gagné")
    # darshan.seFaireAttaquer(hichem)
    # # print("aprés premiere attaque : ", darshan)
    # hichem.seFaireAttaquer(darshan)
    # # print("aprés premiere attaque : ", hichem)
    # darshan.seFaireAttaquer(hichem)
    # # print("aprés deuxieme frappe : ", darshan)
    # hichem.seFaireAttaquer(darshan)
    # # print("aprés deuxieme frappe : ", hichem)
    # darshan.seFaireAttaquer(hichem)
    # # print("aprés troisieme frappe : ", darshan)
    # hichem.seFaireAttaquer(darshan)
    # # print("aprés troisieme frappe : ", hichem)
    # darshan.seFaireAttaquer(hichem)
    # # print("aprés quatrieme frappe : ", darshan)
    # hichem.seFaireAttaquer(darshan)
    # # print("aprés quatrieme frappe : ", hichem)
    # darshan.seFaireAttaquer(hichem)
    # # print("aprés cinquieme frappe : ", darshan)
    # hichem.seFaireAttaquer(darshan)
    # # print("aprés cinquieme frappe : ", hichem)
    # darshan.seFaireAttaquer(hichem)
    # # print("aprés sixieme frappe : ", darshan)
    # hichem.seFaireAttaquer(darshan)
    # # print("aprés sixieme frappe : ", hichem)
    # darshan.seFaireAttaquer(hichem)
    # # print("aprés septieme frappe : ", darshan)
    # hichem.seFaireAttaquer(darshan)
    # # print("aprés septieme frappe : ", hichem)
    # darshan.seFaireAttaquer(hichem)
    # # print("aprés huitieme frappe : ", darshan)
    # hichem.seFaireAttaquer(darshan)
    # # print("aprés huitieme frappe : ", hichem)
    # darshan.seFaireAttaquer(hichem)
    # # print("aprés neuvieme frappe : ", darshan)

# hichem=Humain("hichem", 40, "ingé")
# print("mon humain hichem qui est un objet ",hichem)
# print("l'age DE hichem we access to this information with the object point hichem.",hichem.age)
#
#
# list_fonctionnaire=[]
# print("ma liste sensé vide ",list_fonctionnaire)
# list_fonctionnaire.append(hichem)
# print("mon premier element à lindice 0",list_fonctionnaire[0])
# print("les infos sur hichem", hichem.age, hichem.nom, hichem.fonction)
# print("l'age de hichem", list_fonctionnaire[0].age)
#
# kheira=Humain("afra", 27, "student")
# list_fonctionnaire.append(kheira)
# print("le resultat du calcul * 6 de la fonction calcul de kheira ", kheira.calcul(5))
#
#


# create a list of human
# liste_humain = []
# liste_humain.append(Humain("hichem", 20))
# liste_humain.append(Humain("kheira", 18))
# print(liste_humain[0].nom)
#
# # import matplotlib.pyplot as plt
# import numpy as kheira
# b = np.
# # create numpy 2 dimensional array
# a = np.array([[1, 2, 3], [4, 5, 6]])
# print(a)
# # create a figure for the a array
# fig = plt.scatter([1, 2, 3], [4, 5, 6], c='r')
# # create a subplot for the a array
# # ax = fig.add_subplot(111)
# # ay = fig.add_subplot(10, 1, 2)
# # plot the a array
# # ax.plot(a)
# plt.plot([1, 2, 3], [4, 5, 6], c='r')
# # show the plot
# plt.show()
#
#
#
