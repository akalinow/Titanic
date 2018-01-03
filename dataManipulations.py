import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold

##############################################################################
##############################################################################
##############################################################################
class dataManipulations:

    def getNumpyMatricesFromRawData(self):

        data = pd.read_csv(self.fileName, sep=',',header=0)
        data.replace(to_replace=dict(female=-1, male=1), inplace=True)
        data.replace(to_replace=dict(C=1,Q=2,S=3), inplace=True)
        data.fillna(value=0,inplace=True)

        features = data.values
        labels = features[:,1]
        features = np.delete(features,1,1) #Delete label column
        features = np.delete(features,2,1) #Drop person name column
        features = np.delete(features,6,1) #Drop ticket number
        features = np.delete(features,0,1) #Drop passager id
        features = np.delete(features,6,1) #Drop cabin number (to be used later)

        print(features[0])

        assert features.shape[0] == labels.shape[0]

        self.features_placeholder = tf.placeholder(tf.float32)
        self.labels_placeholder = tf.placeholder(tf.float32)
        self.features = features
        self.labels = labels


    def makeCVFoldGenerator(self, nFolds):

        foldSplitter = KFold(n_splits=nFolds)
        self.foldsIndexGenerator = foldSplitter.split(self.labels, self.features)

    def makeDatasets(self):

        aDataset = tf.contrib.data.Dataset.from_tensor_slices((self.features_placeholder, self.labels_placeholder))
        self.trainDataset = aDataset.batch(self.batchSize)

        aDataset = tf.contrib.data.Dataset.from_tensor_slices((self.features_placeholder, self.labels_placeholder))
        self.validationDataset = aDataset.batch(self.batchSize)


    def getDataIteratorAndInitializerOp(self, aDataset):

        aIterator = tf.contrib.data.Iterator.from_structure(aDataset.output_types, aDataset.output_shapes)
        init_op = aIterator.make_initializer(aDataset)
        return aIterator, init_op

    def getCVFold(self, sess, aFold):

        aCounter = 0
        for trainIndexes, validationIndexes in self.foldsIndexGenerator:
            print("aCounter", aCounter)
            if aCounter==aFold:

                #print("validationIndexes",validationIndexes)
                #print("trainIndexes",trainIndexes)
                #print("train features",self.features[trainIndexes[0]])
                #print("validation features",self.features[validationIndexes[0]])

                foldFeatures = self.features[trainIndexes]
                foldLabels = self.labels[trainIndexes]
                feed_dict={self.features_placeholder: foldFeatures, self.labels_placeholder: foldLabels}
                sess.run(self.trainIt_InitOp, feed_dict=feed_dict)

                foldFeatures = self.features[validationIndexes]
                foldLabels = self.labels[validationIndexes]
                feed_dict={self.features_placeholder: foldFeatures, self.labels_placeholder: foldLabels}
                sess.run(self.validationIt_InitOp, feed_dict=feed_dict)

                return self.trainIterator.get_next(), self.validationIterator.get_next()
            else:
                aCounter+=1

        print("Fold too big: ",aFold," number of folds is ",nFolds)
        return None

    def __init__(self, fileName, nFolds, batchSize):
        self.fileName = fileName
        self.batchSize = batchSize

        self.getNumpyMatricesFromRawData()
        self.makeCVFoldGenerator(nFolds)
        self.makeDatasets()

        self.trainIterator, self.trainIt_InitOp = self.getDataIteratorAndInitializerOp(self.trainDataset)
        self.validationIterator, self.validationIt_InitOp = self.getDataIteratorAndInitializerOp(self.validationDataset)

##############################################################################
##############################################################################
##############################################################################
def makeFeedDict(sess, x, y_, dataIter):
    aBatch = sess.run(dataIter)
    xs = aBatch[0]
    ys = np.reshape(aBatch[1],(-1,1))
    return {x: xs, y_: ys}
##############################################################################
##############################################################################
##############################################################################
