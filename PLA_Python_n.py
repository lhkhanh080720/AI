import numpy as np
import matplotlib.pyplot as plt

#Input Data
#...

#Perceptron Learning Algorithm
def PLA(X, Y):
	#X = [W0, W1, W2]
	W = np.zeros(len(X[0]))
	for t in range(20):
		for i in range(len(Y)):
			if Y[i] == 1 and W @ X[i] < 0:
				W += X[i]
			if Y[i] == 0 and W @ X[i] >= 0:
				W -= X[i]
	print("W =", W)

#Main
if __name__ == "__main__":
    PLA(X, Y)