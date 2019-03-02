# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Build the Inception v3 network on ImageNet data set.

The Inception v3 architecture is described in http://arxiv.org/abs/1512.00567

Summary of available functions:
 inference: Compute inference on the model inputs to make a prediction
 loss: Compute the loss of the prediction with respect to the labels
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf

from inception.slim import slim
from inception.slim import ops


FLAGS = tf.app.flags.FLAGS

# If a model is trained using multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

# Batch normalization. Constant governing the exponential moving average of
# the 'global' mean and variance for all activations.
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999


def inference(images, num_classes, for_training=False, restore_logits=True,
              scope=None):
  """Build Inception v3 model architecture.

  See here for reference: http://arxiv.org/abs/1512.00567

  Args:
    images: Images returned from inputs() or distorted_inputs().
    num_classes: number of classes
    for_training: If set to `True`, build the inference model for training.
      Kernels that operate differently for inference during training
      e.g. dropout, are appropriately configured.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: optional prefix string identifying the ImageNet tower.

  Returns:
    Logits. 2-D float Tensor.
    Auxiliary Logits. 2-D float Tensor of side-head. Used for training only.
  """
  # Parameters for BatchNorm.
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
  }
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
    with slim.arg_scope([slim.ops.conv2d],
                        stddev=0.1,
                        activation=tf.nn.relu,
                        batch_norm_params=batch_norm_params):
      logits, endpoints = slim.peta.inception_v3(
          images,
          dropout_keep_prob=0.8,
          num_classes=num_classes,
          is_training=for_training,
          restore_logits=restore_logits,
          scope=scope)

  # Add summaries for viewing model statistics on TensorBoard.
  _activation_summaries(endpoints)

  # Grab the logits associated with the side head. Employed during training.
  auxiliary_logits = endpoints['aux_logits']
  # logits = tf.Print(logits, [ops.flatten(logits)], 'logits= ', summarize=30)
  # auxiliary_logits = tf.Print(auxiliary_logits, [ops.flatten(auxiliary_logits)], 'logits= ', summarize=30)

  return logits, auxiliary_logits


def inference_bn(images, num_classes, for_training=False, restore_logits=True,
              scope=None):
  """Build Inception v3 model architecture.

  See here for reference: http://arxiv.org/abs/1512.00567

  Args:
    images: Images returned from inputs() or distorted_inputs().
    num_classes: number of classes
    for_training: If set to `True`, build the inference model for training.
      Kernels that operate differently for inference during training
      e.g. dropout, are appropriately configured.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: optional prefix string identifying the ImageNet tower.

  Returns:
    Logits. 2-D float Tensor.
    Auxiliary Logits. 2-D float Tensor of side-head. Used for training only.
  """
  # Parameters for BatchNorm.
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
  }
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
    with slim.arg_scope([slim.ops.conv2d],
                        stddev=0.1,
                        activation=tf.nn.relu,
                        batch_norm_params=batch_norm_params):
      logits, endpoints = slim.peta.inception_v3_bn(
          images,
          dropout_keep_prob=0.8,
          num_classes=num_classes,
          is_training=for_training,
          restore_logits=restore_logits,
          scope=scope)

  # Add summaries for viewing model statistics on TensorBoard.
  _activation_summaries(endpoints)

  # Grab the logits associated with the side head. Employed during training.
  auxiliary_logits = endpoints['aux_logits']
  # logits = tf.Print(logits, [ops.flatten(logits)], 'logits= ', summarize=30)
  # auxiliary_logits = tf.Print(auxiliary_logits, [ops.flatten(auxiliary_logits)], 'logits= ', summarize=30)

  return logits, auxiliary_logits

def inference_roi_head(images, rois,  num_classes, for_training=False, restore_logits=True,
              scope=None):
  """Build Inception v3 model architecture.

  See here for reference: http://arxiv.org/abs/1512.00567

  Args:
    images: Images returned from inputs() or distorted_inputs().
    num_classes: number of classes
    for_training: If set to `True`, build the inference model for training.
      Kernels that operate differently for inference during training
      e.g. dropout, are appropriately configured.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: optional prefix string identifying the ImageNet tower.

  Returns:
    Logits. 2-D float Tensor.
    Auxiliary Logits. 2-D float Tensor of side-head. Used for training only.
  """
  # Parameters for BatchNorm.
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
  }
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
    with slim.arg_scope([slim.ops.conv2d],
                        stddev=0.1,
                        activation=tf.nn.relu,
                        batch_norm_params=batch_norm_params):
      logits, endpoints = slim.peta.inception_v3_roi_head(
          images,
          rois,
          dropout_keep_prob=0.8,
          num_classes=num_classes,
          is_training=for_training,
          restore_logits=restore_logits,
          scope=scope)

  # Add summaries for viewing model statistics on TensorBoard.
  _activation_summaries(endpoints)

  # Grab the logits associated with the side head. Employed during training.
  auxiliary_logits = endpoints['aux_logits']
  # logits = tf.Print(logits, [ops.flatten(logits)], 'logits= ', summarize=30)
  # auxiliary_logits = tf.Print(auxiliary_logits, [ops.flatten(auxiliary_logits)], 'logits= ', summarize=30)

  return logits, auxiliary_logits


def inference_lstm(images, rois, num_classes, for_training=False, restore_logits=True,
              scope=None):
  """Build Inception v3 model architecture.

  See here for reference: http://arxiv.org/abs/1512.00567

  Args:
    images: Images returned from inputs() or distorted_inputs().
    num_classes: number of classes
    for_training: If set to `True`, build the inference model for training.
      Kernels that operate differently for inference during training
      e.g. dropout, are appropriately configured.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: optional prefix string identifying the ImageNet tower.

  Returns:
    Logits. 2-D float Tensor.
    Auxiliary Logits. 2-D float Tensor of side-head. Used for training only.
  """
  # Parameters for BatchNorm.
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
  }
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
    with slim.arg_scope([slim.ops.conv2d],
                        stddev=0.1,
                        activation=tf.nn.relu,
                        batch_norm_params=batch_norm_params):
      logits, endpoints = slim.peta.inception_v3_lstm(
          images,
          rois,
          dropout_keep_prob=0.8,
          num_classes=num_classes,
          is_training=for_training,
          restore_logits=restore_logits,
          scope=scope)

  # Add summaries for viewing model statistics on TensorBoard.
  _activation_summaries(endpoints)

  # Grab the logits associated with the side head. Employed during training.
  auxiliary_logits = endpoints['aux_logits']
  # logits = tf.Print(logits, [ops.flatten(logits)], 'logits= ', summarize=30)
  # auxiliary_logits = tf.Print(auxiliary_logits, [ops.flatten(auxiliary_logits)], 'logits= ', summarize=30)

  return logits, auxiliary_logits

def inference_roi_lstm(images, rois, num_classes, for_training=False, restore_logits=True,
              scope=None):
  """Build Inception v3 model architecture.

  See here for reference: http://arxiv.org/abs/1512.00567

  Args:
    images: Images returned from inputs() or distorted_inputs().
    num_classes: number of classes
    for_training: If set to `True`, build the inference model for training.
      Kernels that operate differently for inference during training
      e.g. dropout, are appropriately configured.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: optional prefix string identifying the ImageNet tower.

  Returns:
    Logits. 2-D float Tensor.
    Auxiliary Logits. 2-D float Tensor of side-head. Used for training only.
  """
  # Parameters for BatchNorm.
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
  }
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
    with slim.arg_scope([slim.ops.conv2d],
                        stddev=0.1,
                        activation=tf.nn.relu,
                        batch_norm_params=batch_norm_params):
      logits, endpoints = slim.peta.inception_v3_roi_lstm(
          images,
          rois,
          dropout_keep_prob=0.8,
          num_classes=num_classes,
          is_training=for_training,
          restore_logits=restore_logits,
          scope=scope)

  # Add summaries for viewing model statistics on TensorBoard.
  _activation_summaries(endpoints)

  # Grab the logits associated with the side head. Employed during training.
  auxiliary_logits = endpoints['aux_logits']
  # logits = tf.Print(logits, [ops.flatten(logits)], 'logits= ', summarize=30)
  # auxiliary_logits = tf.Print(auxiliary_logits, [ops.flatten(auxiliary_logits)], 'logits= ', summarize=30)

  return logits, auxiliary_logits

def inference_roi_lstm_bn(images, rois, num_classes, for_training=False, restore_logits=True,
              scope=None):
  """Build Inception v3 model architecture.

  See here for reference: http://arxiv.org/abs/1512.00567

  Args:
    images: Images returned from inputs() or distorted_inputs().
    num_classes: number of classes
    for_training: If set to `True`, build the inference model for training.
      Kernels that operate differently for inference during training
      e.g. dropout, are appropriately configured.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: optional prefix string identifying the ImageNet tower.

  Returns:
    Logits. 2-D float Tensor.
    Auxiliary Logits. 2-D float Tensor of side-head. Used for training only.
  """
  # Parameters for BatchNorm.
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
  }
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
    with slim.arg_scope([slim.ops.conv2d],
                        stddev=0.1,
                        activation=tf.nn.relu,
                        batch_norm_params=batch_norm_params):
      logits, endpoints = slim.peta.inception_v3_roi_lstm_bn(
          images,
          rois,
          dropout_keep_prob=0.8,
          num_classes=num_classes,
          is_training=for_training,
          restore_logits=restore_logits,
          scope=scope)

  # Add summaries for viewing model statistics on TensorBoard.
  _activation_summaries(endpoints)

  # Grab the logits associated with the side head. Employed during training.
  auxiliary_logits = endpoints['aux_logits']
  # logits = tf.Print(logits, [ops.flatten(logits)], 'logits= ', summarize=30)
  # auxiliary_logits = tf.Print(auxiliary_logits, [ops.flatten(auxiliary_logits)], 'logits= ', summarize=30)

  return logits, auxiliary_logits


def inference_roi_lstm_loc2glo(images, rois, num_classes, for_training=False, restore_logits=True,
              scope=None):
  """Build Inception v3 model architecture.

  See here for reference: http://arxiv.org/abs/1512.00567

  Args:
    images: Images returned from inputs() or distorted_inputs().
    num_classes: number of classes
    for_training: If set to `True`, build the inference model for training.
      Kernels that operate differently for inference during training
      e.g. dropout, are appropriately configured.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: optional prefix string identifying the ImageNet tower.

  Returns:
    Logits. 2-D float Tensor.
    Auxiliary Logits. 2-D float Tensor of side-head. Used for training only.
  """
  # Parameters for BatchNorm.
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
  }
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
    with slim.arg_scope([slim.ops.conv2d],
                        stddev=0.1,
                        activation=tf.nn.relu,
                        batch_norm_params=batch_norm_params):
      logits, endpoints = slim.peta.inception_v3_roi_lstm_loc2glo(
          images,
          rois,
          dropout_keep_prob=0.8,
          num_classes=num_classes,
          is_training=for_training,
          restore_logits=restore_logits,
          scope=scope)

  # Add summaries for viewing model statistics on TensorBoard.
  _activation_summaries(endpoints)

  # Grab the logits associated with the side head. Employed during training.
  auxiliary_logits = endpoints['aux_logits']
  # logits = tf.Print(logits, [ops.flatten(logits)], 'logits= ', summarize=30)
  # auxiliary_logits = tf.Print(auxiliary_logits, [ops.flatten(auxiliary_logits)], 'logits= ', summarize=30)

  return logits, auxiliary_logits

def inference_roi_lstm_loc2glo_bn(images, rois, num_classes, for_training=False, restore_logits=True,
              scope=None):
  """Build Inception v3 model architecture.

  See here for reference: http://arxiv.org/abs/1512.00567

  Args:
    images: Images returned from inputs() or distorted_inputs().
    num_classes: number of classes
    for_training: If set to `True`, build the inference model for training.
      Kernels that operate differently for inference during training
      e.g. dropout, are appropriately configured.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: optional prefix string identifying the ImageNet tower.

  Returns:
    Logits. 2-D float Tensor.
    Auxiliary Logits. 2-D float Tensor of side-head. Used for training only.
  """
  # Parameters for BatchNorm.
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
  }
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
    with slim.arg_scope([slim.ops.conv2d],
                        stddev=0.1,
                        activation=tf.nn.relu,
                        batch_norm_params=batch_norm_params):
      logits, endpoints = slim.peta.inception_v3_roi_lstm_loc2glo_bn(
          images,
          rois,
          dropout_keep_prob=0.8,
          num_classes=num_classes,
          is_training=for_training,
          restore_logits=restore_logits,
          scope=scope)

  # Add summaries for viewing model statistics on TensorBoard.
  _activation_summaries(endpoints)

  # Grab the logits associated with the side head. Employed during training.
  auxiliary_logits = endpoints['aux_logits']
  # logits = tf.Print(logits, [ops.flatten(logits)], 'logits= ', summarize=30)
  # auxiliary_logits = tf.Print(auxiliary_logits, [ops.flatten(auxiliary_logits)], 'logits= ', summarize=30)

  return logits, auxiliary_logits

def loss(logits, labels, batch_size=None):
  """Adds all losses for the model.

  Note the final loss is not returned. Instead, the list of losses are collected
  by slim.losses. The losses are accumulated in tower_loss() and summed to
  calculate the total loss.

  Args:
    logits: List of logits from inference(). Each entry is a 2-D float Tensor.
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
    batch_size: integer
  """
  if not batch_size:
    batch_size = FLAGS.batch_size

  # Reshape the labels into a dense Tensor of
  # shape [FLAGS.batch_size, num_classes].
  # sparse_labels = tf.reshape(labels, [batch_size, 30])
  # indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
  # concated = tf.concat(axis=1, values=[indices, sparse_labels])
  num_classes = logits[0].get_shape()[-1].value
  # dense_labels = tf.sparse_to_dense(concated,
  #                                  [batch_size, num_classes],
  #                                  1.0, 0.0)
  # print (num_classes)
  # print(logits[0])
  # label_weight = [0.3049, 0.0102, 0.4294, 0.5604, 0.0043, 0.1884, 0.9451, 0.017, 0.073, 0.0105, 0.2213, 0.1238, 0.0582, 0.2289, 0.1715, 0.3151, 0.0226, 0.0309, 0.0176, 0.5624, 0.065, 0.065, 0.065, 0.267, 0.1326, 0.3024, 0.2718, 0.1236, 0.0124, 0.1572]
  # label_weight=[0.3049, 0.0102, 0.4294, 0.5604, 0.1453, 0.7437, 0.0963, 0.9466, 0.0478, 0.0043, 0.1884, 0.9451, 0.017, 0.073, 0.0105, 0.2213, 0.1238, 0.0582, 0.2289, 0.1715, 0.3151, 0.0226, 0.0309, 0.0176, 0.5624, 0.065, 0.065, 0.065, 0.267, 0.1326, 0.3024, 0.2718, 0.1236, 0.0124, 0.1572, 0.0179, 0.0638, 0.028, 0.0421, 0.0272, 0.0124, 0.0238, 0.293, 0.0338, 0.0329, 0.0916, 0.0246, 0.0101, 0.017, 0.0229, 0.1278]
  label_weight = [0.5004, 0.3231, 0.1031, 0.0639, 0.1984, 0.1976, 0.8607, 0.8520, 0.1380, 0.1343, 0.1016, 0.0700, 0.3058, 0.2945, 0.0395, 0.2394, 0.5511, 0.2932, 0.0816, 0.7465, 0.2757, 0.0265, 0.0774, 0.0199, 0.3627, 0.0336, 0.1440, 0.0443, 0.2198, 0.0163, 0.0317, 0.5201, 0.0842, 0.4526, 0.0136]
  # Cross entropy loss for the main softmax prediction.
  slim.losses.weighted_sigmoid_cross_entropy_loss(logits[0],
                                 labels,
                                 label_weight, 
                                 label_smoothing=0.1,
                                 weight=1.0)

  # Cross entropy loss for the auxiliary softmax head.
  slim.losses.weighted_sigmoid_cross_entropy_loss(logits[1],
                                 labels,
                                 label_weight,
                                 label_smoothing=0.1,
                                 weight=0.4,
                                 scope='aux_loss')


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _activation_summaries(endpoints):
  with tf.name_scope('summaries'):
    for act in endpoints.values():
      _activation_summary(act)
