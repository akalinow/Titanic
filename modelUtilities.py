import tensorflow as tf
import numpy as np

##############################################################################
##############################################################################
##############################################################################
def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    #initial = tf.truncated_normal(shape, stddev=0.1)
    numberOfInputs = shape[0]
    initial = tf.random_uniform(shape)*np.sqrt(2.0/numberOfInputs)
    return tf.Variable(initial)
##############################################################################
##############################################################################
##############################################################################
def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)
##############################################################################
##############################################################################
##############################################################################
def variable_summaries(var):
    return #temporary swithc off
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)
##############################################################################
##############################################################################
##############################################################################
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        #tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      #tf.summary.histogram('activations', activations)
      return activations
##############################################################################
##############################################################################
##############################################################################
def defineNetwork():
    nHiddenNeurons = [128, 64, 32, 16]
    nLayers = 4

  with tf.device('/cpu:0'):
      aHiddenLayer = nn_layer(x, nInputFeatures, nHiddenNeurons[0], 'hidden0',act=tf.nn.relu)
      myLayers = [aHiddenLayer]
      for iLayer in range(1,nLayers):
          previousLayer = myLayers[iLayer-1]
          aHiddenLayer = nn_layer(previousLayer, nHiddenNeurons[iLayer-1], nHiddenNeurons[iLayer], 'hidden'+str(iLayer),act=tf.nn.relu)
          myLayers.append(aHiddenLayer)

      lastLayer = myLayers[nLayers-1]

      with tf.name_scope('dropout'):
         keep_prob = tf.placeholder(tf.float32)
         #tf.summary.scalar('dropout_keep_probability', keep_prob)
         dropped = tf.nn.dropout(lastLayer, keep_prob)

      # Do not apply softmax activation yet, see below.
      y = nn_layer(dropped, nHiddenNeurons[nLayers-1], 1, 'output', act=tf.identity)