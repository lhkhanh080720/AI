from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

#Input
Weight = np.array([338, 428, 352, 374, 450, 462, 332, 338, 444, 388, 408, 388, 360,
                   570, 492, 496, 462, 436, 432, 464, 418, 442, 408, 428, 558, 512,
                   490, 480, 512, 460, 484, 528, 498, 442, 610, 746, 678, 574, 684,
                   638, 602, 600, 616, 584, 482, 454, 484, 484, 440, 388
])
Area = np.array([230.3, 234.06, 220.6, 227.7, 251.07, 245.4, 214.1, 208.2, 238.5, 231.8, 216.77, 231.08, 214.1,
                 274.25, 253.58, 244.89, 248.10, 237.16, 213.26, 250.66, 235.76, 234.7, 224.6, 232.06, 270.7, 268.3,
                 242.9, 252.6, 255.2, 233.91, 248.4, 254.2, 242, 239.9, 292.1, 309.8, 296.2, 265.95, 308.1,
                 296, 273.7, 266, 273.3, 285.3, 251.35, 240.08, 242.99, 246.3, 230.1, 224.29
])

#Linear Regression (Simple Regression)
def Linear_Regression(X, Y):
    w1 = ((X*Y).sum() - (X.sum())*(Y.sum())/len(X)) / ((X*X).sum() - pow((X.sum()),2)/len(X))
    w0 = (Y.sum() - w1*(X.sum()))/len(X)
    W = np.array([w0, w1])
    return W
    
#Plot
def plot_point(X, Y, W):
    for i in range(len(Y)):
        plt.scatter(X[i], Y[i], s=40, marker='.', color='red')
    sampleX = np.linspace(min(X[:]) - 5, max(X[:] + 5), int((max(X[:]) - min(X[:]) + 10)*100))
    sampleY = W[0] + W[1]*sampleX
    plt.plot(sampleX, sampleY, color='black')
    plt.xlabel("Area(cm2)")
    plt.ylabel("Weight(g)")
    plt.show()


#Main
if __name__ == "__main__":
    W = Linear_Regression(Area, Weight)
    plot_point(Area, Weight, W)
