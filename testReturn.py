def hichem(a,b):
    z = a + b
    return z

def useReturn(z):
    y = z + 2
    return y

def main():
    print(hichem(1,2))

if __name__ == '__main__':
    # main()
    hichem(3,4)
    a = hichem(4,6)
    print(hichem(3,4))
    # print(z)
    print(useReturn(hichem(3,4)))
    b = useReturn(a)
    print(b)

