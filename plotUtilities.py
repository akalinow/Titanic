import numpy as np
import matplotlib.pyplot as plt


def plotHistogram(xSurvive, xDied, xAll, nBins, normed, range):

    # the histogram of the data
    nPass, bins, patches = plt.hist(x, bins = nBins, range = range, normed = normed, facecolor='g', alpha=0.75)
    nNorm, bins, patches = plt.hist(norm, bins = nBins, range = range, normed = normed, facecolor='g', alpha=0.75)

    print(nPass)
    print(nNorm)

    nPass = nPass/nNorm
    nPass = np.nan_to_num(nPass)

    plt.hist(nPass, bins = nBins, range = range, normed = normed, facecolor='g', alpha=0.75)


    print("n: ",nPass)
    plt.xlabel('Feature')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.show()


def plotVariable(iVariable, x, y):

    y = np.broadcast_to(y,x.shape)

    survivedIndexes = y[:,0]==1.0
    diedIndexes = y[:,0]==0.0

    survivedFeatures = x[survivedIndexes]
    diedFeaturses = x[diedIndexes]

    print(survivedFeatures)

    plotHistogram(x = survivedFeatures[:,0], norm = x[:,0], nBins = 21, normed = 0,range=(0,4))
    #plotHistogram(x = survivedFeatures[:,1], nBins = 21, normed = 0,range=(-2,2))
    #plotHistogram(x = survivedFeatures[:,2], nBins = 20, normed = 0,range=(0,100))
    #plotHistogram(x = survivedFeatures[:,3], nBins = 10, normed = 0,range=(0,10))
    #plotHistogram(x = survivedFeatures[:,4], nBins = 10, normed = 0,range=(0,10))
    #plotHistogram(x = survivedFeatures[:,5], nBins = 10, normed = 0,range=(0,100))
    #plotHistogram(x = survivedFeatures[:,6], nBins = 5, normed = 0,range=(0,5))
