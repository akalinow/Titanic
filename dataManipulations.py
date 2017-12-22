import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold

##############################################################################
##############################################################################
##############################################################################
def getNumpyMatricesFromRawData(fileName):

    data = pd.read_csv(fileName, sep=',',header=0)
    data.replace(to_replace=dict(female=-1, male=1), inplace=True)
    data.replace(to_replace=dict(C=1,Q=2,S=3), inplace=True)
    data.fillna(value=0,inplace=True)

    features = data.values
    labels = features[:,1]
    #Delete label column
    features = np.delete(features,1,1)
    #Drop person name column
    features = np.delete(features,2,1)
    #Drop ticket number
    features = np.delete(features,6,1)
    #Drop passager id
    features = np.delete(features,0,1)
    #Drop cabin number (to be used later)
    features = np.delete(features,6,1)

    assert features.shape[0] == labels.shape[0]

    return features, labels
##############################################################################
##############################################################################
##############################################################################
def makeDataIteratorAndInitializerOp(aDataset):

    aIterator = tf.contrib.data.Iterator.from_structure(aDataset.output_types, aDataset.output_shapes)
    init_op = aIterator.make_initializer(aDataset)

    return aIterator, init_op
##############################################################################
##############################################################################
##############################################################################
def makeDataset(rawFeatures, rawLabels):

        features_placeholder = tf.placeholder(tf.float32, rawFeatures.shape)
        labels_placeholder = tf.placeholder(tf.float32, rawLabels.shape)
        dataset = tf.contrib.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
        return dataset
##############################################################################
##############################################################################
##############################################################################
def getCVFoldGenerator(nFolds, labels, features):

    foldSplitter = KFold(n_splits=nFolds)
    foldsIndexGenerator = foldSplitter.split(labels, features)

    return foldsIndexGenerator
##############################################################################
##############################################################################
##############################################################################
def getCVFold(aFold, nFolds, labels, features):

    foldsIndexGenerator = getCVFoldGenerator(nFolds, features, labels)

    aCounter = 0
    for trainIndexes, testIndexes in foldsIndexGenerator:
        print("aCounter", aCounter)
        if aCounter==aFold:

            print("testIndexes",testIndexes)
            aFeatures = features[trainIndexes]
            aLabels = labels[trainIndexes]
            trainDataset = makeDataset(features, labels)

            aFeatures = features[testIndexes]
            aLabels = labels[testIndexes]
            testDataset = makeDataset(features, labels)

            return trainDataset, testDataset
        else:
             aCounter+=1

    print("Fold too big: ",aFold," number of folds is ",nFolds)
    return None

##############################################################################
##############################################################################
##############################################################################
def getTrainAndTestFoldDatasets(aFold, nFolds, fileName):

    features, labels = getNumpyMatricesFromRawData(fileName="data/train/train.csv")
    featuresDataset, labelsDataset = getCVFold(aFold, nFolds, features, labels)
    return featuresDataset, labelsDataset
##############################################################################
##############################################################################
##############################################################################
def makeFeedDict(sess, x, y_, dataIter):
    aBatch = sess.run(dataIter)
    xs = aBatch[0]
    ys = np.reshape(aBatch[1],(-1,1))
    k = 1.0
    return {x: xs, y_: ys}
##############################################################################
##############################################################################
##############################################################################
