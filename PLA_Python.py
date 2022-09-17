from os import X_OK
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

#Input
def Read_Data():
	df = pd.read_csv('D:\HCMUT\AI\HW\PLA.csv')
	plt.plot(df['X1'][1:12], df['X2'][1:12], 'go', color='green')
	plt.plot(df['X1'][12:], df['X2'][12:], 'gs', color='red')
	plt.xlabel('X1') # label x 
	plt.ylabel('X2') # label y 
	plt.show()
	return df

#Perceptron Learning Algorithm
def PLA(data):
	X = []
	for i in range(0, 22):
		X.append([1, data['X1'][i], data['X2'][i]])
	W = np.zeros(len(X[0]))
	for i in range(400):
		for i in range(len(X)):
			if data['Class'][i] == 1 and (W @ np.array(X[i])) < 0:
				W += X[i]
			if data['Class'][i] == 0 and (W @ np.array(X[i])) >= 0:
				W -= X[i]
	print("W =", W)
	sampleX = np.linspace(min(data['X1']) - 3, max(data['X1']) + 3, int((max(data['X1']) - min(data['X1']))*100))
	sampleY = (-W[0] - W[1]*sampleX)/W[2]
	plt.plot(sampleX, sampleY, color='black')
	plt.plot(data['X1'][1:12], data['X2'][1:12], 'go', color='green')
	plt.plot(data['X1'][12:], data['X2'][12:], 'gs', color='red')
	plt.xlabel('X1') # label x 
	plt.ylabel('X2') # label y 
	plt.show()
	return W

#Main
if __name__ == "__main__":
	data = Read_Data()
	W = PLA(data)
	x_test = []
	for i in range(5):
		x_test.append([1, random.randint(0, 11), random.randint(0, 11)])
		plt.plot(x_test[i][1], x_test[i][2], 'b^', color='black')
	plt.plot(data['X1'][1:12], data['X2'][1:12], 'go', color='green')
	plt.plot(data['X1'][12:], data['X2'][12:], 'gs', color='red')
	sampleX = np.linspace(min(data['X1']) - 3, max(data['X1']) + 3, int((max(data['X1']) - min(data['X1']))*100))
	sampleY = (-W[0] - W[1]*sampleX)/W[2]
	plt.plot(sampleX, sampleY, color='black')
	plt.xlabel('X1') # label x 
	plt.ylabel('X2') # label y 
	plt.show()

	for i in range(len(x_test)):
		if W @ np.array(x_test[i]) < 0:
			plt.plot(x_test[i][1], x_test[i][2], 'b^', color='red')
		else:
			plt.plot(x_test[i][1], x_test[i][2], 'b^', color='green')
		circle1 = plt.Circle((x_test[i][1], x_test[i][2]), .2, color='black', fill = False)
		plt.gcf().gca().add_artist(circle1)
	plt.plot(data['X1'][1:12], data['X2'][1:12], 'go', color='green')
	plt.plot(data['X1'][12:], data['X2'][12:], 'gs', color='red')
	sampleX = np.linspace(min(data['X1']) - 3, max(data['X1']) + 3, int((max(data['X1']) - min(data['X1']))*100))
	sampleY = (-W[0] - W[1]*sampleX)/W[2]
	plt.plot(sampleX, sampleY, color='black')
	plt.xlabel('X1') # label x 
	plt.ylabel('X2') # label y 
	plt.show()
	
