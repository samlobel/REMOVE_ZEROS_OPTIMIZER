import tensorflow as tf
import numpy as np


LR=0.01
BATCH_SIZE=8
NUM_MINIBATCHES=10000

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
from time import time

from models import four_layer_relu, two_layer_relu

import remove_zeros_utils


print('creating models')

# model_dict = two_layer_relu(BATCH_SIZE)
model_dict = four_layer_relu(BATCH_SIZE)


print('creating updates')

updates = remove_zeros_utils.create_consistency_update_for_list_of_tensors(
  model_dict['CE_UNPACKED'],
  model_dict['VAR_LIST'],
  LR
)

consistency_for_W1 = remove_zeros_utils.get_consistency_from_error(
  model_dict['CE_UNPACKED'], model_dict['VAR_LIST'][0]
)



# updates = tf.train.GradientDescentOptimizer(LR).minimize(model_dict['CE_REDUCED'], var_list=model_dict['VAR_LIST'])
# updates = tf.train.MomentumOptimizer(LR, 0.5).minimize(model_dict['CE_REDUCED'], var_list=model_dict['VAR_LIST'])
# updates = tf.train.AdamOptimizer(1e-3).minimize(model_dict['CE_REDUCED'], var_list=model_dict['VAR_LIST'])


print('updates created')

def get_and_print_consistency(sess, feed_dict):
  print('method called')
  _consistency = sess.run(consistency_for_W1, feed_dict=feed_dict)
  # print('consistency:')
  # print(_consistency)
  print('consistency shape: {}'.format(_consistency.shape))
  print('middle: {}'.format(_consistency[28*14 + 14]))


errors = []
if __name__ == '__main__':
  init = tf.initialize_all_variables()
  sess = tf.Session()
  sess.run(init)

  for i in range(NUM_MINIBATCHES):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    feed_dict={
      model_dict['INPUT_PH'] : batch_xs,
      model_dict['OUTPUT_PH']: batch_ys
    }
    get_and_print_consistency(sess, feed_dict)    
    _, _cross_entropy = sess.run([updates, model_dict['CE_REDUCED']], feed_dict=feed_dict)
    # correct_prediction = tf.equal(tf.argmax(model_dict['OUTPUT_PH_TEST'],1), tf.argmax(model_dict['OUTPUT_CALC_TEST'],1))
    if i % 10 == 0:
      acc, _ce = sess.run([model_dict['ACCURACY_TEST'], model_dict['CE_TEST']], feed_dict={
        model_dict['INPUT_PH_TEST']: mnist.test.images,
        model_dict['OUTPUT_PH_TEST']: mnist.test.labels
      })
      print('BATCH: {}      ACCURACY: {}     CE: {}'.format(i, acc, _ce))




