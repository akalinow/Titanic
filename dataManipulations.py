import numpy as np
import pandas as pd
import tensorflow as tf

##############################################################################
##############################################################################
##############################################################################
def readData(sess, fileName, batch):

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

        features_placeholder = tf.placeholder(tf.float32, features.shape)
        labels_placeholder = tf.placeholder(tf.float32, labels.shape)

        dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
        #dataset = dataset.repeat()
        dataset = dataset.batch(batch)

        iterator = dataset.make_initializable_iterator()

        sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                          labels_placeholder: labels})

        numberOfFeatures = features.shape[1]
        return iterator.get_next(), numberOfFeatures
##############################################################################
##############################################################################
##############################################################################
def makeFeedDict(sess, x, y_, dataIter):
    aBatch = sess.run(dataIter)
    xs = aBatch[0]
    ys = np.reshape(aBatch[1],(-1,1))
    k = 1.0
    return {x: xs, y_: ys}
