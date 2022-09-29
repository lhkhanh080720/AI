#hard margin SVM, soft margin SVM
from turtle import title
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_blobs
from sklearn.svm import SVC

def initData(samples = 100, cluster = 0.5):
    #init datapoints
    X, y = make_blobs(n_samples=samples, centers=2, cluster_std=cluster, random_state=0)
    
    for i, val in enumerate(X):
        if y[i]:
            plt.scatter(X[i, 0], X[i, 1], s=40, color = 'g') 
        else:
            plt.scatter(X[i, 0], X[i, 1], s=40, color = 'red')

    plt.xlabel('X0')
    plt.ylabel('X1')
    plt.show()

    return (X, y)

def SVM(X, y, type = None, C = None):        
    if type == 'hard':                                  #hard margin SVM
        model = SVC(kernel='linear')                    # Create hyperplane: linear
    elif type == 'soft':                                #soft margin SVM 
        model = SVC(kernel='linear', C = C)             # Create hyperplane: linear                   
    model.fit(X, y)                              
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

    #Circle Point
    for i in range(len(model.support_vectors_[:, 0])):
        circle1 = plt.Circle((model.support_vectors_[i, 0], model.support_vectors_[i, 1]), .1, color='black', fill = False)
        plt.gcf().gca().add_artist(circle1)
    ax = plt.gca()
    
    #Border
    x = np.linspace(min(X[:, 0]) - 0.5, max(X[:, 0]) + 0.5)
    y = np.linspace(min(X[:, 1]) - 0.51, max(X[:, 1]) + 0.5)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    ax.axis("equal")
    plt.show()

def processImage():
    #Load Image
    img = plt.imread('image.png')
    plt.imshow(img)
    
    #Sample Selection
    samp = [img[260:285, 164:189], img[2:27, 2:27]]
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.imshow(samp[i])
        if i:
            plt.gca().set_title('Water')
        else:
            plt.gca().set_title('Soil')
    plt.show()

    data = []
    target = []
    for i in range(2):
        for j in range(25):
            for k in range(25):
                data += [samp[i][j][k]]
                target += [i]

    #SVM
    svm = SVC(C = 1)
    svm.fit(data, target)
    ROW,COL,RGB = np.shape(img)
    gambar = img.reshape(ROW*COL, RGB)
    kelas = svm.predict(gambar)
    kelas = kelas.reshape(ROW,COL)
    output = np.zeros((2, ROW, COL, RGB))
    for i in range(ROW):
        for j in range(COL):
            output[kelas[i][j]][i][j] = img[i][j]

    plt.subplot(3,2,(1,3)).set_title('Image')
    plt.imshow(img)
    plt.subplot(3,2,2).set_title('Sample Soil')
    plt.imshow(samp[0])
    plt.subplot(3,2,4).set_title('Sample Water')
    plt.imshow(samp[1])

    titleOut = ['Soil', 'Water']
    for i in range(2):
        plt.subplot(3,2,i+5).set_title(titleOut[i])
        plt.imshow(output[i])

    plt.subplots_adjust(bottom=0.05, left=0.05, top = 0.975, right=0.95)
    plt.show()

if __name__ == '__main__':
    input = input("Input = ")
    if input == 'hard':
        (X, y) = initData(150, 0.5)
        SVM(X, y, type='hard')
    elif input == 'soft':
        (X, y) = initData(90, 1)
        SVM(X, y, type='soft', C=55)
    else:
        processImage()