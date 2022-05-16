import numpy as np

a=0
for i in range(1,10):
    print("i : ",i)
    a+=i
    print("a : ", a)
    b = np.sum(a)
    print("b : ",b)