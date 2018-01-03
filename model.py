from modelUtilites import *


deviceName = '/cpu:0'
##############################################################################
##############################################################################
##############################################################################
def fcModel(x, nNeurons):

    nLayers = len(nNeurons)
    myLayers = [x]

    with tf.device(deviceName):
        for iLayer in range(1,nLayers):
            previousLayer = myLayers[iLayer-1]
            nInputs = nNeurons[iLayer-1]
            layerName =  'hidden'+str(iLayer)
            aLayer = nn_layer(previousLayer, nInputs, nNeurons[iLayer], layerName, act=tf.nn.elu)
            myLayers.append(aLayer)

      lastLayer = myLayers[nLayers-1]

      with tf.name_scope('dropout'):
         keep_prob = tf.placeholder(tf.float32)
         #tf.summary.scalar('dropout_keep_probability', keep_prob)
         dropped = tf.nn.dropout(lastLayer, keep_prob)

      # Do not apply softmax activation yet, see below.
      y = nn_layer(lastLayer, nNeurons[nLayers-1], 1, 'output', act=tf.identity)

      with tf.name_scope('cross_entropy'):
          diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
          lambdaLagrange = 0.01
          modelParameters   = tf.trainable_variables()
          lossL2 = tf.add_n([ tf.nn.l2_loss(aParam) for aParam in modelParameters
                         if 'biases' not in aParam.name ]) * lambdaLagrange

      with tf.name_scope('total'):
          cross_entropy = tf.reduce_mean(diff + lossL2)
      tf.summary.scalar('cross_entropy', cross_entropy)

      with tf.name_scope('train'):
          train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

      with tf.name_scope('accuracy'):
          with tf.name_scope('correct_prediction'):
              correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
          accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      tf.summary.scalar('accuracy', accuracy)
      tf.summary.histogram('correct_prediction', correct_prediction)

  # Merge all the summaries and write them out to
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  tf.global_variables_initializer().run()

  builder = tf.saved_model.builder.SavedModelBuilder(FLAGS.model_dir)

  return train_step, builder
##############################################################################
##############################################################################
##############################################################################
