import tensorflow as tf
import numpy as np


def get_expanded_grads(unpacked_error, var):
  return tf.pack([tf.gradients(err, var)[0] for err in unpacked_error])

def count_not_zeros(expanded_grads):
  zeros_tensor = tf.not_equal(expanded_grads, 0.0)
  return tf.reduce_sum(tf.cast(zeros_tensor, tf.float32), reduction_indices=[0])

def get_scaled_not_zeros(expanded_grads):
  count_zeros = count_not_zeros(expanded_grads)
  scaled = count_zeros / tf.reduce_mean(count_zeros)
  return scaled

def count_positives(expanded_grads):
  greater_tensor = tf.greater(expanded_grads, 0.0)
  return tf.reduce_sum(tf.cast(greater_tensor, tf.float32), reduction_indices=[0])

def count_negatives(expanded_grads):
  less_tensor = tf.less(expanded_grads, 0.0)
  return tf.reduce_sum(tf.cast(less_tensor, tf.float32), reduction_indices=[0])

def get_consistency(expanded_grads):
  # num_elements = expanded_grads.get_shape().as_list()[0]
  pos_tensor = count_positives(expanded_grads)
  neg_tensor = count_negatives(expanded_grads)
  # not_zero_tensor = get_scaled_not_zeros(expanded_grads)
  consistency  = tf.abs(pos_tensor - neg_tensor)
  consistency = consistency / tf.reduce_mean(consistency)
  # consistency = consistency  / not_zero_tensor
  # consistency_normed = 
  # consistency = consistency / num_elements
  return consistency

def get_consistency_from_error(unpacked_error, var):
  expanded_grads = get_expanded_grads(unpacked_error, var)
  return get_consistency(expanded_grads)

def create_consistency_update_for_tensor(unpacked_error, var, lr=0.001):
  expanded_grads = get_expanded_grads(unpacked_error, var)
  consistency_scaler = get_consistency(expanded_grads)
  zeros_scaler = get_scaled_not_zeros(expanded_grads)
  grad_mean = tf.reduce_mean(expanded_grads, reduction_indices=[0])
  grad_scaled = tf.mul(grad_mean, consistency_scaler)
  grad_scaled = tf.div(grad_scaled, zeros_scaler)
  update = tf.assign(var, var - lr*grad_scaled)
  return update

def create_consistency_update_for_list_of_tensors(unpacked_error, var_list, lr=0.001):
  update_list = [create_consistency_update_for_tensor(unpacked_error, var, lr) for var in var_list]
  update_group = tf.group(*update_list)
  return update_group

def get_not_zero_count(unpacked_error, var):
  expanded_grads =get_expanded_grads(unpacked_error, var)
  return count_not_zeros(expanded_grads)

def scale_grads_by_zeros(expanded_grads, MIN_ZEROS=1):
  num_zeros_unclipped = count_not_zeros(expanded_grads)
  num_zeros = tf.maximum(num_zeros_unclipped, MIN_ZEROS)
  num_zeros_scaled = tf.div(num_zeros, tf.reduce_mean(num_zeros))
  grad_sum = tf.reduce_sum(expanded_grads, reduction_indices=[0])
  grad_scaled = tf.div(grad_sum, num_zeros)
  return grad_scaled

def create_update_for_tensor(unpacked_error, var, lr=0.001):
  expanded_grads = get_expanded_grads(unpacked_error, var)
  grad_scaled = scale_grads_by_zeros(expanded_grads)
  update = tf.assign(var, var - lr*grad_scaled)
  return update

def create_updates_for_list_of_tensors(unpacked_error, var_list, lr=0.001):
  update_list = [create_update_for_tensor(unpacked_error, var, lr) for var in var_list]
  update_group = tf.group(*update_list)
  return update_group

# def update_average_gradients(unpacked_error, var_list, momentum_list, lr=0.001):
#   update_list = [create_update_for_tensor(unpacked_error, var, lr) for var in var_list]

