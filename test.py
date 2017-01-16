# Looks like I need a test, to make sure that the equal zero thing works.

import tensorflow as tf

import numpy as np


x = tf.placeholder(tf.float32, [10,10])

not_zero_bools = tf.not_equal(x, 0.0)
num_not_zero = tf.reduce_sum(tf.cast(not_zero_bools, tf.float32))




input_one = np.random.rand(10,10)


print(input_one)
for a in np.nditer(input_one, op_flags=['readwrite']):
  a[...] = 0 if a < 0.5 else a
print(input_one)

with tf.Session() as sess:
  num_not_zero = sess.run(num_not_zero, feed_dict={x : input_one})
  print('num not zero: {}'.format(num_not_zero))






