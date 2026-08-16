from matrix import matrix


x = matrix([[i*8+j for j in range(1,9)] for i in range(8)])
x.print()
print("maxpool 2")
x.maxPool(2).print()
print("maxpool 4")
x.maxPool(4).print()
