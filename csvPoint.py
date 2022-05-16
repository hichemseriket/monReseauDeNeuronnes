import csv
import numpy as np

with open('input1.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    # lines = len(list(reader))
    # print("lines",lines)
    colonnes = []
    a=[]
    print("reader",reader)
    # for row in range(1,reader.line_num):
    for row in reader:
        print(" un tableau censé etre toute la ligne  ", row)

        # for i in range(0,row):
        for i in row:

            print("cesnsé etre la case", i)
            colonnes.append(i)
            print("collonne", colonnes)
    a.append(colonnes)
    print("a", a)

