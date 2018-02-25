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

import xgboost as xgb
import numpy as np
from dataManipulations import *
from plotUtilities import *

FLAGS = None

##############################################################################
##############################################################################
##############################################################################
def runCVFold(iFold, myDataManipulations):

    features, labels = myDataManipulations.getCVFoldArrays(iFold, isValidation=False)

    trainData = xgb.DMatrix(features, label=labels)
    # specify parameters via map
    param = {'max_depth':14, 'eta':0.1, 'subsample': 0.25,
             'scale_pos_weight':1.0, 'objective':'binary:logistic',
             'silent': 1,
              'eval_metric': 'error'}

    num_round = FLAGS.max_epoch
    bst = xgb.train(param, trainData, num_round)

    #xgb.plot_importance(bst)

    modelResult = bst.predict(trainData)
    plotDiscriminant(modelResult, labels, "Train")

    features, labels = myDataManipulations.getCVFoldArrays(iFold, isValidation=True)
    testData = xgb.DMatrix(features, label=labels)
    modelResult = bst.predict(testData)
    foldAccuracy = plotDiscriminant(modelResult, labels, "Validation")

    return foldAccuracy

##############################################################################
##############################################################################
##############################################################################
def train():

    nFolds = 2
    nEpochs = FLAGS.max_epoch
    batchSize = 900
    fileName = FLAGS.train_data_file

    myDataManipulations = dataManipulations(fileName, nFolds, nEpochs, batchSize)
    accuracyTable = np.array([])

    for iFold in range(0, nFolds):
        aAccuracy = runCVFold(iFold, myDataManipulations)
        accuracyTable = np.append(accuracyTable, aAccuracy)

    print("Mean accuracy: ",np.mean(accuracyTable),"+-",np.std(accuracyTable,ddof=1))

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
