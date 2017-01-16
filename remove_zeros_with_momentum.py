import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
from time import time


"""
GAMEPLAN:
When you take the unpacked derivative, you count up the number of non-zero-entries for each parameter.
Take max(non_zero, 1): That makes sure you don't divide by zero.. 
Then, you do reduce_sum, and then tf.div(sum, num_zeros). That gives you a scaled average. Should be better.

Once I do that, I think the standard deviation argument makes a lot more sense. It was really thrown
off by the zeros.

I also would like to get a count of the number of zeros for each parameter.
Is that something that's big for any column but the first? Or the last? I don't know.

Also, it would be cool to do it for sparse and then for not sparse.

"""


print('not really done at all')
exit()

NUM_EPOCHS = 10
BATCH_SIZE=64
NUM_MINIBATCHES = 50000 * NUM_EPOCHS // BATCH_SIZE
print('num minibatches: {}'.format(NUM_MINIBATCHES))
LR = 0.05
MIN_ZEROS=1
OPT_METHOD = 'MY_WAY'


def get_expanded_grads(unpacked_error, var):
  return tf.pack([tf.gradients(err, var)[0] for err in unpacked_error])

def count_not_zeros(expanded_grads):
  zeros_tensor = tf.not_equal(expanded_grads, 0.0)
  zeros_counted = tf.reduce_sum(tf.cast(zeros_tensor, tf.float32), reduction_indices=[0])
  stopped_grads = tf.stop_gradient(zeros_counted)
  return stopped_grads

def get_zero_count(unpacked_error, var):
  expanded_grads = get_expanded_grads(unpacked_error, var)
  return count_not_zeros(expanded_grads)

def scale_grads_by_zeros(expanded_grads):
  num_zeros_unclipped = count_not_zeros(expanded_grads)
  num_zeros = tf.maximum(num_zeros_unclipped, MIN_ZEROS)
  grad_sum = tf.reduce_sum(expanded_grads, reduction_indices=[0])
  grad_scaled = tf.div(grad_sum, num_zeros)
  return grad_scaled

def create_update_for_tensor(unpacked_error, var, lr=0.001):
  expanded_grads = get_expanded_grads(unpacked_error, var)
  grad_scaled = scale_grads_by_zeros(expanded_grads)
  update = tf.assign(var, var - lr*grad_scaled)
  return update

def update_momentum_placeholders(update_list, var_list, momentum):
  zipped = zip(update_list, var_list)


def create_updates_for_list_of_tensors(unpacked_error, var_list, momentum_list, lr=0.001, momentum=0.5):
  update_list = [create_update_for_tensor(unpacked_error, var, lr) for var in var_list]
  update_group = tf.group(*update_list)
  return update_group

def create_unscaled_updates(error, var_list, lr=0.001):
  grads = [tf.gradients(error, var)[0] for var in var_list]
  updates = [tf.assign(var, var - (lr*grad)) for (var, grad) in zip(var_list, grads)]
  return tf.group(*updates)



print('making placeholders')
x = tf.placeholder(tf.float32, [BATCH_SIZE, 784])
y_ = tf.placeholder(tf.float32, [BATCH_SIZE, 10])
y_argmax = tf.argmax(y_,1)

print('creating layer one')
W1 = tf.Variable(tf.truncated_normal([784, 200], stddev=0.001))
b1 = tf.Variable(tf.zeros([200]))
h1 = tf.nn.relu(tf.matmul(x,W1) + b1)

print('creating layer two')
W2 = tf.Variable(tf.truncated_normal([200, 10], stddev=0.001))
b2 = tf.Variable(tf.zeros([10]))
unscaled_y = tf.matmul(h1,W2) + b2
y = tf.nn.softmax(unscaled_y)

print('calculating error')
ce_not_reduced = tf.nn.softmax_cross_entropy_with_logits(unscaled_y, y_)
cross_entropy = tf.reduce_mean(ce_not_reduced)
ce_unpacked = tf.unpack(ce_not_reduced)
# y = tf.nn.softmax(tf.matmul(h1,W2) + b2)


START_TIME =  time()

VAR_LIST = [W1,b1,W2,b2]
print('creating updates')
# updates = tf.train.MomentumOptimizer(LR, 0.5).minimize(cross_entropy)
# updates = tf.train.GradientDescentOptimizer(LR).minimize(cross_entropy)
# updates = create_unscaled_updates(cross_entropy, VAR_LIST, LR)
updates = create_updates_for_list_of_tensors(ce_unpacked, VAR_LIST, LR)
print('updates created')

print('SECONDS TO CREATE UPDATES: {}'.format((time() - START_TIME)))
# exit()

# zero_count_b1 = get_zero_count(ce_unpacked, b1)
# zero_count_b2 = get_zero_count(ce_unpacked, b2)
# zero_count_W1 = get_zero_count(ce_unpacked, W1)



# cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# ce_not_reduced = -tf.reduce_sum(y_*tf.log(y), reduction_indices=[1])
# ce_unpacked = tf.unpack(ce_not_reduced)




print('re-calculating everything, so you can use variable inputs...')
xTEST = tf.placeholder(tf.float32, [None, 784])
y_TEST = tf.placeholder(tf.float32, [None, 10])
y_TEST_argmax = tf.argmax(y_TEST,1)

h1TEST = tf.nn.relu(tf.matmul(xTEST,W1) + b1)
unscaled_yTEST = tf.matmul(h1TEST,W2) + b2
yTEST = tf.nn.softmax(unscaled_yTEST)

ce_not_reduced_TEST = tf.nn.softmax_cross_entropy_with_logits(unscaled_yTEST, y_TEST)
ce_TEST = tf.reduce_mean(ce_not_reduced_TEST)
# ce_TEST = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(unscaled_yTEST, y_TEST))


# ce_TEST = -tf.reduce_sum(y_TEST*tf.log(yTEST))






# update_scaling = [apply_scaled_grad(cross_entropy, ce_unpacked, v, LR) for v in VAR_LIST]
# update_not_scaling = [apply_regular_grad(cross_entropy, v, LR) for v in VAR_LIST]

# train_scaling = apply_scaled_grad(cross_entropy, ce_unpacked, W1, 0.01)
# train_regular = apply_regular_grad(cross_entropy, W1, 0.01)


errors = []
if __name__ == '__main__':
  init = tf.initialize_all_variables()
  sess = tf.Session()
  sess.run(init)

  for i in range(NUM_MINIBATCHES):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    # _cross_entropy = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
    # _ = sess.run(update_not_scaling, feed_dict={x: batch_xs, y_: batch_ys})
    # print('\r'),
    # print(str(i))

    # _zero_count_W1 = sess.run(zero_count_W1, feed_dict={x : batch_xs, y_: batch_ys})
    # print('non-zero count: Has size: {}\nand value: {}'.format(_zero_count_W1.shape, _zero_count_W1[28*14+14]))
    # _W1 = sess.run(W1, feed_dict={x : batch_xs, y_: batch_ys})
    # print('W1 from middle pixel... count: {}'.format(_W1[28*14+14]))

    _, _cross_entropy = sess.run([updates, cross_entropy], feed_dict={x : batch_xs, y_: batch_ys})
    # print(_cross_entropy)
    # _ = sess.run(unscaled_updates, feed_dict={x : batch_xs, y_: batch_ys})
    # _ = sess.run(update_scaling, feed_dict={x: batch_xs, y_: batch_ys})

    # _, _cross_entropy = sess.run([train_scaling, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
    # _b2_grads, _mean, _stddev, _diff_grads = sess.run([b2_grads, mean, stddev, diff_grads], feed_dict={x: batch_xs, y_: batch_ys})
    # print(_b2_grads)
    # print("MEAN: {}      STDDEV:{}".format(_mean,_stddev))
    # print('diff grads should be around zero... {}'.format(_diff_grads))
    # exit()
    # sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(yTEST,1), tf.argmax(y_TEST,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    if i % 10 == 0:
      # print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
      acc, _ce = sess.run([accuracy, ce_TEST], feed_dict={xTEST: mnist.test.images, y_TEST: mnist.test.labels})
      print("i: {}    CE: {}    ACC: {}".format(i, _ce, acc))
      errors.append(acc)
    if i % 1000 == 0 and i != 0:
      print("\n\n\n")
      print(errors)
      # exit()
      
      # print("i: {}    ACC:   {}\n".format(acc))
      # print(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))








