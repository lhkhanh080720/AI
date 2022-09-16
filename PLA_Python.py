import numpy as np
import matplotlib.pyplot as plt

#Input
#X = [1, X1, X2]
X = np.array([
    [1, -2, 4],
    [1, 4, 1],
    [1, 1, 6],
    [1, 2, 4],
    [1, 6, 2],
])
Y = np.array([0, 0, 1, 1, 1])

#Perceptron Learning Algorithm
def PLA_plot(X, Y):
	#X = [W0, W1, W2]
	W = np.zeros(len(X[0]))
	for t in range(20):
		for i in range(len(Y)):
			if Y[i] == 1 and W @ X[i] < 0:
				W += X[i]
			if Y[i] == 0 and W @ X[i] >= 0:
				W -= X[i]
	print("W =", W)
	sampleX = np.linspace(min(X[:, 1]), max(X[:, 2]), (max(X[:, 2]) - min(X[:, 1]))*100)
	sampleY = (-W[0] - W[1]*sampleX)/W[2]
	plt.plot(sampleX, sampleY, color='black')

#Plot
def plot_point(X, Y):
	for i in range(len(Y)):
		if Y[i] == 0:
			plt.scatter(X[i, 1], X[i, 2], s=40, marker='_', color='red')
		else:
			plt.scatter(X[i, 1], X[i, 2], s=40, marker='+', color='green')
	plt.axis('equal')
	plt.show()

#Main
if __name__ == "__main__":
	PLA_plot(X, Y)
	plot_point(X, Y)
