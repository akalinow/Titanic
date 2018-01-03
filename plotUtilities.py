import numpy as np
import matplotlib.pyplot as plt


def plotHistogram(xSurvive, xDied, xAll, name, nBins, normed, axisRange):

    # the histogram of the data
    fig = plt.figure()
    nPass, bins, patches = plt.hist(xSurvive, bins = nBins, range = axisRange, normed = normed, facecolor='g', alpha=0.75)
    nAll, bins, patches = plt.hist(xAll, bins = nBins, range = axisRange, normed = normed, facecolor='g', alpha=0.75)
    plt.close(fig)


    print("Pass",nPass)
    print("All",nAll)

    nPass = nPass/nAll
    nPass = np.nan_to_num(nPass)
    x = range(0,len(nPass))
    yerr = nPass*(1-nPass)/nAll
    yerr = np.nan_to_num(yerr)
    yerr = np.sqrt(yerr)

    fig = plt.figure(name)
    plt.errorbar(x, nPass, yerr=yerr, fmt='o')

    print("n: ",nPass)
    plt.title(name)
    plt.xlabel('Feature')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.show(block=False)


def plotVariable(iVariable, x, y):

    print (x.shape, y.shape)

    y = np.broadcast_to(y,x.shape)

    survivedIndexes = y[:,0]==1.0
    diedIndexes = y[:,0]==0.0
    allIndexes = y[:,0]>=0.0

    survivedFeatures = x[survivedIndexes]
    diedFeatures = x[diedIndexes]
    allFeatures = x[allIndexes]

    featuresNames = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    featuresRanges = [(0,4), (-2,2), (0,100), (0,10), (0,10), (0,100), (0,5)]
    nFeatures = 6


    for iFeature in range(0, nFeatures+1):
        plotHistogram(xSurvive = survivedFeatures[:,iFeature],
                      xDied = diedFeatures[:,iFeature],
                      xAll = allFeatures[:,iFeature],
                      name = featuresNames[iFeature],
                      nBins = 21, normed = 0,axisRange=featuresRanges[iFeature])
        

    plt.show(block=True)

    #plotHistogram(x = survivedFeatures[:,2], nBins = 20, normed = 0,range=(0,100))
    #plotHistogram(x = survivedFeatures[:,3], nBins = 10, normed = 0,range=(0,10))
    #plotHistogram(x = survivedFeatures[:,4], nBins = 10, normed = 0,range=(0,10))
    #plotHistogram(x = survivedFeatures[:,5], nBins = 10, normed = 0,range=(0,100))
    #plotHistogram(x = survivedFeatures[:,6], nBins = 5, normed = 0,range=(0,5))
