import tensorflow as tf
import numpy as np


LR=0.05
ADAM_LR=1e-3
BATCH_SIZE=32
NUM_MINIBATCHES=10000
# OPT_TYPE = 'MINE'
# OPT_TYPE = 'GRAD_DESC'
# OPT_TYPE = 'MOMENTUM'
OPT_TYPE = 'ADAM'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
from time import time

from models import four_layer_relu, two_layer_relu

import remove_zeros_utils


print('creating models')

# model_dict = two_layer_relu(BATCH_SIZE)
model_dict = four_layer_relu(BATCH_SIZE)





print('creating updates')

if OPT_TYPE=='MINE':
  updates = remove_zeros_utils.create_updates_for_list_of_tensors(
    model_dict['CE_UNPACKED'],
    model_dict['VAR_LIST'],
    LR
  )
if OPT_TYPE=='GRAD_DESC':
  updates = tf.train.GradientDescentOptimizer(LR).minimize(model_dict['CE_REDUCED'], var_list=model_dict['VAR_LIST'])
if OPT_TYPE=='MOMENTUM':
  updates = tf.train.MomentumOptimizer(LR, 0.1).minimize(model_dict['CE_REDUCED'], var_list=model_dict['VAR_LIST'])
if OPT_TYPE=='ADAM':
  updates = tf.train.AdamOptimizer(ADAM_LR).minimize(model_dict['CE_REDUCED'], var_list=model_dict['VAR_LIST'])


print('updates created')

errors = []



if __name__ == '__main__':
  init = tf.initialize_all_variables()
  sess = tf.Session()
  sess.run(init)

  for i in range(NUM_MINIBATCHES):
    if i % 500 == 0:
      acc, _ce = sess.run([model_dict['ACCURACY_TEST'], model_dict['CE_TEST']], feed_dict={
        model_dict['INPUT_PH_TEST']: mnist.test.images,
        model_dict['OUTPUT_PH_TEST']: mnist.test.labels
      })
      print('TYPE: {}      BATCH: {}      ACCURACY: {}     CE: {}'.format(OPT_TYPE, i, acc, _ce))

    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    feed_dict={
      model_dict['INPUT_PH'] : batch_xs,
      model_dict['OUTPUT_PH']: batch_ys
    }

    _, _cross_entropy = sess.run([updates, model_dict['CE_REDUCED']], feed_dict=feed_dict)
    # correct_prediction = tf.equal(tf.argmax(model_dict['OUTPUT_PH_TEST'],1), tf.argmax(model_dict['OUTPUT_CALC_TEST'],1))




