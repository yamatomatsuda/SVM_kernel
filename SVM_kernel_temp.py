import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


# Our dataset and targets
X = np.loadtxt(fname="data1.csv",dtype='float',skiprows=5,usecols=(4,7),delimiter=',')
Y = np.loadtxt(fname="data1.csv",dtype= 'unicode',skiprows=5,usecols=(1),delimiter=',')


for _ in range(len(Y)):
    if '雨' in Y[_]:
        Y[_] = 1
    
    else:
        Y[_] = -1
Y = Y.astype(float)

# figure number
fignum = 1

# fit the model
for kernel in ('linear', 'poly', 'rbf'):
    clf = svm.SVC(kernel=kernel, gamma=2)
    clf.fit(X, Y)

    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum, figsize=(4, 3))
    plt.clf()

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k')
    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
                edgecolors='k')

    plt.axis('tight')

    x_min = 30
    x_max = 100
    y_min = 0
    y_max = 40

    
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    print(Z)
    plt.figure(fignum, figsize=(4, 3))
    #plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())
    fignum = fignum + 1
    
plt.show()