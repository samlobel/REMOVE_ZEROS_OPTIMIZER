import tensorflow as tf
import numpy as np


def two_layer_relu(BATCH_SIZE):
  x = tf.placeholder(tf.float32, [BATCH_SIZE, 784])
  y_ = tf.placeholder(tf.float32, [BATCH_SIZE, 10])

  W1 = tf.get_variable("W1", shape=[784, 200], initializer=tf.contrib.layers.xavier_initializer())
  b1 = tf.Variable(tf.zeros([200]))
  h1 = tf.nn.relu(tf.matmul(x,W1) + b1)

  W2 = tf.get_variable("W4", shape=[200, 10], initializer=tf.contrib.layers.xavier_initializer())
  b2 = tf.Variable(tf.zeros([10]))
  unscaled_y = tf.matmul(h1,W2) + b2
  y = tf.nn.softmax(unscaled_y)

  print('calculating error')
  ce_not_reduced = tf.nn.softmax_cross_entropy_with_logits(unscaled_y, y_)
  cross_entropy = tf.reduce_mean(ce_not_reduced)
  ce_unpacked = tf.unpack(ce_not_reduced)

  print('calculating again, for testing')
  xTEST = tf.placeholder(tf.float32, [None, 784])
  y_TEST = tf.placeholder(tf.float32, [None, 10])

  h1TEST = tf.nn.relu(tf.matmul(xTEST,W1) + b1)
  unscaled_yTEST = tf.matmul(h1TEST,W2) + b2
  yTEST = tf.nn.softmax(unscaled_yTEST)

  ce_not_reduced_TEST = tf.nn.softmax_cross_entropy_with_logits(unscaled_yTEST, y_TEST)
  ce_TEST = tf.reduce_mean(ce_not_reduced_TEST)

  correct_prediction = tf.equal(tf.argmax(yTEST,1), tf.argmax(y_TEST,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  

  print('collecting var list')
  VAR_LIST = [W1,b1,W2,b2]

  return {
    'VAR_LIST' : VAR_LIST,
    'INPUT_PH' : x,
    'OUTPUT_PH' : y_,
    'OUTPUT_CALC' : y,
    'CE_REDUCED' : cross_entropy,
    'CE_UNPACKED' : ce_unpacked,
    'INPUT_PH_TEST' : xTEST,
    'OUTPUT_PH_TEST' : y_TEST,
    'OUTPUT_CALC_TEST' : yTEST,
    'ACCURACY_TEST' : accuracy,
    'CE_TEST' : ce_TEST
  }

def four_layer_relu(BATCH_SIZE):
  x = tf.placeholder(tf.float32, [BATCH_SIZE, 784])
  y_ = tf.placeholder(tf.float32, [BATCH_SIZE, 10])

  W1 = tf.get_variable("W1", shape=[784, 200], initializer=tf.contrib.layers.xavier_initializer())
  b1 = tf.Variable(tf.zeros([200]))
  h1 = tf.nn.relu(tf.matmul(x,W1) + b1)

  W2 = tf.get_variable("W2", shape=[200, 200], initializer=tf.contrib.layers.xavier_initializer())
  b2 = tf.Variable(tf.zeros([200]))
  h2 = tf.nn.relu(tf.matmul(h1,W2) + b2)

  W3 = tf.get_variable("W3", shape=[200, 200], initializer=tf.contrib.layers.xavier_initializer())
  b3 = tf.Variable(tf.zeros([200]))
  h3 = tf.nn.relu(tf.matmul(h2,W3) + b3)  

  W4 = tf.get_variable("W4", shape=[200, 10], initializer=tf.contrib.layers.xavier_initializer())
  b4 = tf.Variable(tf.zeros([10]))
  unscaled_y = tf.matmul(h3,W4) + b4
  y = tf.nn.softmax(unscaled_y)

  print('calculating error')
  ce_not_reduced = tf.nn.softmax_cross_entropy_with_logits(unscaled_y, y_)
  cross_entropy = tf.reduce_mean(ce_not_reduced)
  ce_unpacked = tf.unpack(ce_not_reduced)

  print('calculating again, for testing')
  xTEST = tf.placeholder(tf.float32, [None, 784])
  y_TEST = tf.placeholder(tf.float32, [None, 10])

  h1TEST = tf.nn.relu(tf.matmul(xTEST,W1) + b1)
  h2TEST = tf.nn.relu(tf.matmul(h1TEST,W2) + b2)
  h3TEST = tf.nn.relu(tf.matmul(h2TEST,W3) + b3)
  unscaled_yTEST = tf.matmul(h3TEST,W4) + b4
  yTEST = tf.nn.softmax(unscaled_yTEST)

  ce_not_reduced_TEST = tf.nn.softmax_cross_entropy_with_logits(unscaled_yTEST, y_TEST)
  ce_TEST = tf.reduce_mean(ce_not_reduced_TEST)

  correct_prediction = tf.equal(tf.argmax(yTEST,1), tf.argmax(y_TEST,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  

  print('collecting var list')
  VAR_LIST = [W1,b1,W2,b2,W3,b3,W4,b4]
  # VAR_LIST = [W1,b1,W2,b2] #THIS ONE IS BAD!
  # VAR_LIST = [W3,b3,W4,b4] #THIS ONE IS BAD TOO!

  return {
    'VAR_LIST' : VAR_LIST,
    'INPUT_PH' : x,
    'OUTPUT_PH' : y_,
    'OUTPUT_CALC' : y,
    'CE_REDUCED' : cross_entropy,
    'CE_UNPACKED' : ce_unpacked,
    'INPUT_PH_TEST' : xTEST,
    'OUTPUT_PH_TEST' : y_TEST,
    'OUTPUT_CALC_TEST' : yTEST,
    'ACCURACY_TEST' : accuracy,
    'CE_TEST' : ce_TEST
  }


# def four_layer_relu(BATCH_SIZE):
#   x = tf.placeholder(tf.float32, [BATCH_SIZE, 784])
#   y_ = tf.placeholder(tf.float32, [BATCH_SIZE, 10])

#   W1 = tf.Variable(tf.truncated_normal([784, 200], stddev=0.001))
#   b1 = tf.Variable(tf.zeros([200]))
#   h1 = tf.nn.relu(tf.matmul(x,W1) + b1)

#   W2 = tf.Variable(tf.truncated_normal([200, 200], stddev=0.001))
#   b2 = tf.Variable(tf.zeros([200]))
#   h2 = tf.nn.relu(tf.matmul(h1,W2) + b2)

#   W3 = tf.Variable(tf.truncated_normal([200, 200], stddev=0.001))
#   b3 = tf.Variable(tf.zeros([200]))
#   h3 = tf.nn.relu(tf.matmul(h2,W3) + b3)

#   W4 = tf.Variable(tf.truncated_normal([200, 10], stddev=0.001))
#   b4 = tf.Variable(tf.zeros([10]))
#   unscaled_y = tf.matmul(h3,W4) + b4
#   y = tf.nn.softmax(unscaled_y)

#   print('calculating error')
#   ce_not_reduced = tf.nn.softmax_cross_entropy_with_logits(unscaled_y, y_)
#   cross_entropy = tf.reduce_mean(ce_not_reduced)
#   ce_unpacked = tf.unpack(ce_not_reduced)

#   VAR_LIST = [W1,b1,W2,b2, W3, b3, W4, b4]

#   return {
#     'VAR_LIST' : VAR_LIST,
#     'INPUT_PH' : x,
#     'OUTPUT_PH' : y_,
#     'OUTPUT_CALC' : y,
#     'CE_REDUCED' : cross_entropy,
#     'CE_UNPACKED' : ce_unpacked
#   }

  



