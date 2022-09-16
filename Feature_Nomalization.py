import numpy as np
import matplotlib.pyplot as plt

#Input Data


#Feature Nomalization
def FN(X):
    maxX = max(X[:, 0])
    minX = min(X[:, 0])
    for i, item in enumerate(X):
        item[0] = round(float((item[0] - minX) / (maxX - minX)), 4)
        print(item[0])
    return X

#Plot
def plot_point(X, Y):
	for i in range(len(Y)):
		if Y[i] == "M":
			plt.scatter(X[i, 0], X[i, 1], s=40, marker='_', color='red')
		else:
			plt.scatter(X[i, 0], X[i, 1], s=40, marker='+', color='green')
	plt.axis('equal')
	plt.show()

#Main
if __name__ == "__main__":
    # X = FN(X)
    # plot_point(X, Y)