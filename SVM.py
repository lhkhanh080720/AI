#SVM, soft margin SVM, kernel SVM
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_blobs

#init datapoints
def initData(): 
    #cluster_std: do lech chuan
    X, y = make_blobs(n_samples=150, centers=2, cluster_std=0.5, random_state=0)
    
    for i, val in enumerate(X):
        if y[i]:
            plt.scatter(X[i, 0], X[i, 1], s=40, color = 'g') 
        else:
            plt.scatter(X[i, 0], X[i, 1], s=40, color = 'red')

    plt.xlabel('X0')
    plt.ylabel('X1')
    plt.show()
    
    return (X, y)

#SVM:
def SVM(X, y):
    from sklearn.svm import SVC
    model = SVC(kernel='linear')                        # Create hyperplane: linear
    model.fit(X, y)                                     # Train data
    w = model.coef_[0]             
    a = -w[0]/w[1]
    xx = np.linspace(min(X[:, 0]) - 0.5, max(X[:, 0]) + 0.5)
    yy = a*xx - (model.intercept_[0]/w[1])              # b = model.intercept_[0]
    plt.plot(xx, yy, color='black')

    for i, val in enumerate(X):
        if y[i]:
            plt.scatter(X[i, 0], X[i, 1], s=40, color = 'g') 
        else:
            plt.scatter(X[i, 0], X[i, 1], s=40, color = 'red')

    plt.xlabel('X0')
    plt.ylabel('X1')

    # plot support vectors
    for i in range(len(model.support_vectors_[:, 0])):
        circle1 = plt.Circle((model.support_vectors_[i, 0], model.support_vectors_[i, 1]), .1, color='black', fill = False)
        plt.gcf().gca().add_artist(circle1)
    ax = plt.gca()
    
    # create grid to evaluate model
    x = np.linspace(min(X[:, 0]) - 0.5, max(X[:, 0]) + 0.5)
    y = np.linspace(min(X[:, 1]) - 0.5, max(X[:, 1]) + 0.5)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    ax.axis("equal")
    plt.show()

if __name__ == '__main__':
    (X, y) = initData()
    SVM(X, y)