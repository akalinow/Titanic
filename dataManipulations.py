import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn import preprocessing

##############################################################################
##############################################################################
##############################################################################
class dataManipulations:

    def getNumpyMatricesFromRawData(self):

        data = pd.read_csv(self.fileName, sep=',',header=0)
        data.replace(to_replace=dict(female=-1, male=1), inplace=True)
        data.replace(to_replace=dict(C=2,Q=1,S=5), inplace=True)
        data.fillna(value=0,inplace=True)

        features = data.values
        labels = features[:,1]
        features = np.delete(features,1,1) #Delete label column
        features = np.delete(features,2,1) #Drop person name column
        features = np.delete(features,6,1) #Drop ticket number
        features = np.delete(features,0,1) #Drop passager id

        ##Conver cabin letter to number
        cabinId = features[:,6]
        for counter, value in enumerate(cabinId):
            tmp = str(value)[0]
            cabinId[counter] = ord(tmp)-65
        features[:,6] = cabinId

        #features = np.delete(features,6,1) #Drop cabin number (to be used later)

        #x = np.array(features[:,2])
        #np.log(x)
        #age = np.log(age)
        #age = np.sqrt(age)
        #["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]
        '''
        features[:,0] = 0
        features[:,2] = 0
        features[:,3] = 0
        features[:,4] = 0
        features[:,5] = 0
        features[:,6] = 0
        features[:,7] = 0
        '''
        #print(x)

        min_max_scaler = preprocessing.MinMaxScaler()
        features = min_max_scaler.fit_transform(features)

        #index = features[:,1]==0 #Female
        #index = features[:,1]==1 #Male
        #features = features[index]
        #labels = labels[index]

        #index = features[:,0]==0 #Class
        #features = features[index]
        #labels = labels[index]

        #index = features[:,5]<1 #Fare
        #features = features[index]
        #labels = labels[index]

        #index = features[:,2]<1 #Age
        #features = features[index]
        #labels = labels[index]

        assert features.shape[0] == labels.shape[0]

        self.features_placeholder = tf.placeholder(tf.float32)
        self.labels_placeholder = tf.placeholder(tf.float32)
        self.features = features
        self.labels = labels

    def makeCVFoldGenerator(self):

        foldSplitter = KFold(n_splits=self.nFolds)
        self.foldsIndexGenerator = foldSplitter.split(self.labels, self.features)
        self.indexList = list(enumerate(self.foldsIndexGenerator))

    def makeDatasets(self):

        aDataset = tf.contrib.data.Dataset.from_tensor_slices((self.features_placeholder, self.labels_placeholder))
        self.trainDataset = aDataset.batch(self.batchSize)
        self.trainDataset = self.trainDataset.repeat(self.nEpochs)

        aDataset = tf.contrib.data.Dataset.from_tensor_slices((self.features_placeholder, self.labels_placeholder))
        self.validationDataset = aDataset.batch(self.batchSize)


    def getDataIteratorAndInitializerOp(self, aDataset):

        aIterator = tf.contrib.data.Iterator.from_structure(aDataset.output_types, aDataset.output_shapes)
        init_op = aIterator.make_initializer(aDataset)
        return aIterator, init_op

    def getCVFoldArrays(self, aFold, isValidation):

        indexes = self.indexList[aFold][1][1]
        if isValidation:
            indexes = self.indexList[aFold][1][0]

        features = self.features[indexes]
        labels = self.labels[indexes]

        return features, labels


    def getCVFold(self, sess, aFold):

        if(aFold>=len(self.indexList)):
            print("Fold too big: ",aFold," number of folds is ",self.nFolds)
            return None

        #trainIndexes = self.indexList[aFold][1][0]
        #validationIndexes = self.indexList[aFold][1][1]

        trainIndexes = self.indexList[aFold][1][1]
        validationIndexes = self.indexList[aFold][1][0]

        self.numberOfBatches = np.ceil(len(trainIndexes)/self.batchSize)
        self.numberOfBatches = (int)(self.numberOfBatches)

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

    def __init__(self, fileName, nFolds, nEpochs, batchSize):
        self.fileName = fileName
        self.batchSize = batchSize
        self.nFolds = nFolds
        self.nEpochs = nEpochs

        self.getNumpyMatricesFromRawData()
        self.makeCVFoldGenerator()
        self.makeDatasets()

        self.trainIterator, self.trainIt_InitOp = self.getDataIteratorAndInitializerOp(self.trainDataset)
        self.validationIterator, self.validationIt_InitOp = self.getDataIteratorAndInitializerOp(self.validationDataset)

##############################################################################
##############################################################################
##############################################################################
def makeFeedDict(sess, dataIter):
    aBatch = sess.run(dataIter)
    x = aBatch[0]
    y = np.reshape(aBatch[1],(-1,1))
    return x, y
##############################################################################
##############################################################################
##############################################################################
