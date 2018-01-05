from modelUtilities import *

##############################################################################
##############################################################################
##############################################################################
class Model:

    def addFCLayers(self):

        for iLayer in range(1,self.nLayers):
            previousLayer = self.myLayers[iLayer-1]
            nInputs = self.nNeurons[iLayer-1]
            layerName =  'hidden'+str(iLayer)
            aLayer = nn_layer(previousLayer, nInputs, self.nNeurons[iLayer], layerName, act=tf.nn.elu)
            self.myLayers.append(aLayer)

    def addDropoutLayer(self):

        lastLayer = self.myLayers[-1]

        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            #tf.summary.scalar('dropout_keep_probability', keep_prob)
            aLayer = tf.nn.dropout(lastLayer, keep_prob)
            self.myLayers.append(aLayer)

    def addOutputLayer(self):

        lastLayer = self.myLayers[-1]
        #Do not apply softmax activation yet as softmax is calculated during cross enetropy evaluation
        aLayer = nn_layer(lastLayer, self.nNeurons[self.nLayers-1], 1, 'output', act=tf.identity)
        self.myLayers.append(aLayer)

    def defineOptimizationStrategy(self):

        with tf.name_scope('cross_entropy'):
            y = self.myLayers[-1]
            diff = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.yTrue, logits=y)
            cross_entropy = tf.reduce_mean(diff)
        with tf.name_scope('lossL2'):
            modelParameters   = tf.trainable_variables()
            lossL2 = tf.add_n([ tf.nn.l2_loss(aParam) for aParam in modelParameters
            if 'biases' not in aParam.name ]) * self.lambdaLagrange

        with tf.name_scope('train'):
            lossFunction = tf.add(cross_entropy, lossL2)
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(lossFunction)

        with tf.name_scope('accuracy'):
            y = tf.nn.sigmoid(self.myLayers[-1])
            y = tf.greater(y,0.5)
            y = tf.cast(y, tf.float32)
            correct_prediction = tf.equal(y, self.yTrue)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('cross_entropy', cross_entropy)
        tf.summary.scalar('accuracy', accuracy)

    def __init__(self, x, yTrue, nNeurons, learning_rate, lambdaLagrange):

        self.nNeurons = nNeurons
        self.nLayers = len(self.nNeurons)
        self.myLayers = [x]
        self.yTrue = yTrue

        self.learning_rate = learning_rate
        self.lambdaLagrange = lambdaLagrange

        self.addFCLayers()
        self.addDropoutLayer()
        self.addOutputLayer()
        self.defineOptimizationStrategy()
##############################################################################
##############################################################################
##############################################################################
