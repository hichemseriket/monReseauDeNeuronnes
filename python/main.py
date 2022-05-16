# import numpy as np
# a = [[4,5,6],
#      [7,8,9]]
# b = [[1,2,3],
#      [4,5,6]]
# # a = [[4,5,6],
# #      7,8,9]
# # b = [[1,2,3],
# #      4,5,6]
# # c = []
# #
# # for i in range(len(a)):
# #     for j in range(len(b)):
# #         c.append(a[i]+b[j])
# #
# # print(c)
#
# # create tableau ligne et colonne
# ligne = len(a)
# print(ligne)
# colonne = len(b)
# print(colonne)
# c = [[0 for i in range(colonne)] for j in range(ligne)]
# for i in range(ligne):
#     for j in range(colonne):
#         c[i][j] = a[i][j] + b[i][j]
#         print("les a ",a[i][j])
#         print("les b",b[i][j])
# print(c)
# d = np.array(a)
# print("le nparray d = ",d)
# a = [[1, 2, 3, 4], [5, 6], [7, 8, 9]]
# for row in a:
#     for elem in row:
#         # print(elem, end=' ')
#         print("le tableau a : ",row, "ligne et : ",elem, "colonne" )

# a =[[1,2,3],[4,5,6]]
# for row in a:
#     for elem in row:
#         print(elem, end=' ')
#
# print("le tableau possede : ",row, "ligne et : ",elem, "colonne" )
#
# la première ligne d&#39;entrée est le nombre de lignes du tableau
# print("quel est le nombre de ligne ? : ")
# n = int(input())
# a = []
# for i in range(n):
#     print("entrez la valeur de la ligne : ",i)
#     # a.append(int(input()))
#     row = input().split()
#     for i in range(len(row)):
#         row[i] = int(row[i])
#     a.append(row)
# print(a)

# n = 4
# a = [[0] * n for i in range(n)]
# for i in range(n):
#     for j in range(n):
#         if i < j:
#             a[i][j] = 0
#         elif i > j:
#             a[i][j] = 2
#         else:
#             a[i][j] = 1
# for row in a:
#     print(' '.join([str(elem) for elem in row]))

n = 4
a = [[0] * n for i in range(n)]
for i in range(n):
    for j in range(0, i):
        a[i][j] = 2
    a[i][i] = 1
    for j in range(i + 1, n):
        a[i][j] = 0
for row in a:
    print(' '.join([str(elem) for elem in row]))