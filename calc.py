import sys

kW = [1.33, 1.8, 2.27, 3.03, 5.61, 6.78, 8.04]
# A = [50, 50, 10, 10, 0, 0, 0]
A = eval(sys.argv[1])
print('%d units, %.2f kW' % (sum(A), sum([a * b for (a, b) in zip(kW, A)])))
