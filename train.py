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

FLAGS = None

deviceName = '/cpu:0'
##############################################################################
##############################################################################
##############################################################################
def train():

  sess = tf.Session()

  nFolds = 10
  batchSize = 64
  fileName = FLAGS.train_data_file

  myDataManipulations = dataManipulations(fileName, nFolds, batchSize)

  # Input placeholders
  with tf.name_scope('input'), tf.device(deviceName):
    x = tf.placeholder(tf.float32, name='x-input')
    y_ = tf.placeholder(tf.float32, name='y-input')


  init = tf.global_variables_initializer()
  sess.run(init)

  for aFold in range(0, nFolds):
      aTrainIterator, aValidationIterator = myDataManipulations.getCVFold(sess, aFold)


  for i in range(FLAGS.max_steps):
    if i % 5000 == 0:  # Record summaries and test-set accuracy
      # Record execution stats
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      sess.run([train_step],
               feed_dict=feed_dict(True),
               options=run_options,
               run_metadata=run_metadata)

      result = sess.run([correct_prediction,accuracy,cross_entropy], feed_dict=feed_dict(False))
      sigma = np.nanstd(result[0])
      print('Accuracy at step %s: %s' % (i, result[1]), "Loss: ",result[2])
      print("\tMin/Max deviation: ",np.amin(result[0]), np.amax(result[0]), "sigma: ",sigma)

    else:  # Record a summary
      sess.run([train_step], feed_dict=feed_dict(True))

  train_writer.close()
  # Save the model to disk.
  # Add a second MetaGraphDef for inference.
  builder.add_meta_graph_and_variables(sess,[tf.saved_model.tag_constants.SERVING])
  builder.save()
  print("Model saved in file: %s" % FLAGS.model_dir)







  aResult = sess.run([x,y_],feed_dict=makeFeedDict(sess, x, y_, aTrainIterator))
  labels = aResult[1]
  features = aResult[0]
  plotVariable(0,features, labels)

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
