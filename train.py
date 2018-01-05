#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import numpy as np
from dataManipulations import *
from plotUtilities import *
from model import *

FLAGS = None

deviceName = '/cpu:0'
##############################################################################
##############################################################################
##############################################################################
def runCVFold(sess, iFold, myDataManipulations):
    #Fetch operations
    x = tf.get_default_graph().get_operation_by_name("input/x-input").outputs[0]
    y = tf.get_default_graph().get_operation_by_name("model/output/activation").outputs[0]
    y = tf.nn.softmax(y)
    yTrue = tf.get_default_graph().get_operation_by_name("input/y-input").outputs[0]
    keep_prob = tf.get_default_graph().get_operation_by_name("model/dropout/Placeholder").outputs[0]

    train_step = tf.get_default_graph().get_operation_by_name("model/train/Adam")
    accuracy = tf.get_default_graph().get_operation_by_name("model/accuracy/Mean").outputs[0]
    cross_entropy = tf.get_default_graph().get_operation_by_name("model/cross_entropy/Mean").outputs[0]
    #cross_entropy = tf.get_default_graph().get_operation_by_name("model/cross_entropy/SoftmaxCrossEntropyWithLogits").inputs[1]
    lossL2 = tf.get_default_graph().get_operation_by_name("model/lossL2/mul").outputs[0]

    #mergedSummary = tf.get_default_graph().get_operation_by_name("monitor/Merge/MergeSummary").outputs[0]

    aTrainIterator, aValidationIterator = myDataManipulations.getCVFold(sess, iFold)
    foldLoss = 0

    #Train
    for iEpoch in range(0,FLAGS.max_epoch):
        while True:
            try:
                xs, ys = makeFeedDict(sess, aTrainIterator)
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                sess.run([train_step],
                feed_dict={x: xs, yTrue: ys, keep_prob: 1.0},
                options=run_options,
                run_metadata=run_metadata)
                ########################################
                #Evaluate training perormance
                result = sess.run([accuracy, cross_entropy, lossL2, y, yTrue],
                feed_dict={x: xs, yTrue: ys, keep_prob: 1.0})
            except tf.errors.OutOfRangeError:
                break
        #Evaluate training perormance
        result = sess.run([accuracy, cross_entropy, lossL2, y, yTrue],
        feed_dict={x: xs, yTrue: ys, keep_prob: FLAGS.dropout})
        if(iEpoch%10==0):
            print("Epoch: ",iEpoch,
            "Accuracy: ", result[0],
            "cross entropy: ",result[1],
            "L2 loss: ",result[2])
    #########################################
    #Evaluate performance on validation data
    while True:
        try:
            xs, ys = makeFeedDict(sess, aValidationIterator)
            result = sess.run([cross_entropy, lossL2],
            feed_dict={x: xs, yTrue: ys, keep_prob: 1.0})
            foldLoss = result[0] + result[1]
            ########################################
        except tf.errors.OutOfRangeError:
            break

    return foldLoss
##############################################################################
##############################################################################
##############################################################################
def train():

    sess = tf.Session()

    nFolds = 10
    batchSize = 64
    fileName = FLAGS.train_data_file
    nNeurons = [7,10]

    myDataManipulations = dataManipulations(fileName, nFolds, batchSize)

    # Input placeholders
    with tf.name_scope('input'), tf.device(deviceName):
        x = tf.placeholder(tf.float32, name='x-input')
        yTrue = tf.placeholder(tf.float32, name='y-input')

    with tf.name_scope('model'), tf.device(deviceName):
        myModel = Model(x, yTrue, nNeurons, FLAGS.learning_rate, FLAGS.lambda_lagrange)

    init = tf.global_variables_initializer()
    sess.run(init)
    # Merge all the summaries and write them out to
    with tf.name_scope('monitor'), tf.device(deviceName):
        merged = tf.summary.merge_all()
    myWriter = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    builder = tf.saved_model.builder.SavedModelBuilder(FLAGS.model_dir)
    ###############################################
    ops = tf.get_default_graph().get_operations()
    for op in ops:
        print(op.name)


    meanLoss = 0

    for iFold in range(0, nFolds):
        meanLoss += runCVFold(sess, iFold, myDataManipulations)

    meanLoss/=nFolds
    print("Mean loss for ",nFolds," folds: ",meanLoss)

    myWriter.close()
    # Save the model to disk.
    # Add a second MetaGraphDef for inference.
    builder.add_meta_graph_and_variables(sess,[tf.saved_model.tag_constants.SERVING])
    builder.save()
    print("Model saved in file: %s" % FLAGS.model_dir)
    '''
    aResult = sess.run([x,y_],feed_dict=makeFeedDict(sess, x, y_, aTrainIterator))
    labels = aResult[1]
    features = aResult[0]
    plotVariable(0,features, labels)
    '''
    return
##############################################################################
##############################################################################
##############################################################################
def main(_):

  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)

  if tf.gfile.Exists(FLAGS.model_dir):
    tf.gfile.DeleteRecursively(FLAGS.model_dir)

  train()
##############################################################################
##############################################################################
##############################################################################
if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--max_epoch', type=int, default=50,
                      help='Number of epochs')

  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')

  parser.add_argument('--lambda_lagrange', type=float, default=0.01,
                      help='Largange multipler for L2 loss')

  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')

  parser.add_argument('--train_data_file', type=str,
      default=os.path.join(os.getenv('PWD', './'),
                           'data/train/train.csv'),
      help='Directory for storing training data')

  parser.add_argument('--model_dir', type=str,
      default=os.path.join(os.getenv('PWD', './'),
                           'model'),
      help='Directory for storing model state')

  parser.add_argument('--log_dir', type=str,
      default=os.path.join(os.getenv('PWD', './'),
                           'logs'),
      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
