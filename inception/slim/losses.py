# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains convenience wrappers for various Neural Network TensorFlow losses.

  All the losses defined here add themselves to the LOSSES_COLLECTION
  collection.

  l1_loss: Define a L1 Loss, useful for regularization, i.e. lasso.
  l2_loss: Define a L2 Loss, useful for regularization, i.e. weight decay.
  cross_entropy_loss: Define a cross entropy loss using
    softmax_cross_entropy_with_logits. Useful for classification.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# In order to gather all losses in a network, the user should use this
# key for get_collection, i.e:
#   losses = tf.get_collection(slim.losses.LOSSES_COLLECTION)
LOSSES_COLLECTION = '_losses'


def l1_regularizer(weight=1.0, scope=None):
  """Define a L1 regularizer.

  Args:
    weight: scale the loss by this factor.
    scope: Optional scope for name_scope.

  Returns:
    a regularizer function.
  """
  def regularizer(tensor):
    with tf.name_scope(scope, 'L1Regularizer', [tensor]):
      l1_weight = tf.convert_to_tensor(weight,
                                       dtype=tensor.dtype.base_dtype,
                                       name='weight')
      return tf.multiply(l1_weight, tf.reduce_sum(tf.abs(tensor)), name='value')
  return regularizer


def l2_regularizer(weight=1.0, scope=None):
  """Define a L2 regularizer.

  Args:
    weight: scale the loss by this factor.
    scope: Optional scope for name_scope.

  Returns:
    a regularizer function.
  """
  def regularizer(tensor):
    with tf.name_scope(scope, 'L2Regularizer', [tensor]):
      l2_weight = tf.convert_to_tensor(weight,
                                       dtype=tensor.dtype.base_dtype,
                                       name='weight')
      return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
  return regularizer


def l1_l2_regularizer(weight_l1=1.0, weight_l2=1.0, scope=None):
  """Define a L1L2 regularizer.

  Args:
    weight_l1: scale the L1 loss by this factor.
    weight_l2: scale the L2 loss by this factor.
    scope: Optional scope for name_scope.

  Returns:
    a regularizer function.
  """
  def regularizer(tensor):
    with tf.name_scope(scope, 'L1L2Regularizer', [tensor]):
      weight_l1_t = tf.convert_to_tensor(weight_l1,
                                         dtype=tensor.dtype.base_dtype,
                                         name='weight_l1')
      weight_l2_t = tf.convert_to_tensor(weight_l2,
                                         dtype=tensor.dtype.base_dtype,
                                         name='weight_l2')
      reg_l1 = tf.multiply(weight_l1_t, tf.reduce_sum(tf.abs(tensor)),
                      name='value_l1')
      reg_l2 = tf.multiply(weight_l2_t, tf.nn.l2_loss(tensor),
                      name='value_l2')
      return tf.add(reg_l1, reg_l2, name='value')
  return regularizer


def l1_loss(tensor, weight=1.0, scope=None):
  """Define a L1Loss, useful for regularize, i.e. lasso.

  Args:
    tensor: tensor to regularize.
    weight: scale the loss by this factor.
    scope: Optional scope for name_scope.

  Returns:
    the L1 loss op.
  """
  with tf.name_scope(scope, 'L1Loss', [tensor]):
    weight = tf.convert_to_tensor(weight,
                                  dtype=tensor.dtype.base_dtype,
                                  name='loss_weight')
    loss = tf.multiply(weight, tf.reduce_sum(tf.abs(tensor)), name='value')
    tf.add_to_collection(LOSSES_COLLECTION, loss)
    return loss


def l2_loss(tensor, weight=1.0, scope=None):
  """Define a L2Loss, useful for regularize, i.e. weight decay.

  Args:
    tensor: tensor to regularize.
    weight: an optional weight to modulate the loss.
    scope: Optional scope for name_scope.

  Returns:
    the L2 loss op.
  """
  with tf.name_scope(scope, 'L2Loss', [tensor]):
    weight = tf.convert_to_tensor(weight,
                                  dtype=tensor.dtype.base_dtype,
                                  name='loss_weight')
    loss = tf.multiply(weight, tf.nn.l2_loss(tensor), name='value')
    tf.add_to_collection(LOSSES_COLLECTION, loss)
    return loss


def cross_entropy_loss(logits, one_hot_labels, label_smoothing=0,
                       weight=1.0, scope=None):
  """Define a Cross Entropy loss using softmax_cross_entropy_with_logits.

  It can scale the loss by weight factor, and smooth the labels.

  Args:
    logits: [batch_size, num_classes] logits outputs of the network .
    one_hot_labels: [batch_size, num_classes] target one_hot_encoded labels.
    label_smoothing: if greater than 0 then smooth the labels.
    weight: scale the loss by this factor.
    scope: Optional scope for name_scope.

  Returns:
    A tensor with the softmax_cross_entropy loss.
  """
  logits.get_shape().assert_is_compatible_with(one_hot_labels.get_shape())
  with tf.name_scope(scope, 'CrossEntropyLoss', [logits, one_hot_labels]):
    num_classes = one_hot_labels.get_shape()[-1].value
    one_hot_labels = tf.cast(one_hot_labels, logits.dtype)
    if label_smoothing > 0:
      smooth_positives = 1.0 - label_smoothing
      smooth_negatives = label_smoothing / num_classes
      one_hot_labels = one_hot_labels * smooth_positives + smooth_negatives
    cross_entropy = tf.contrib.nn.deprecated_flipped_softmax_cross_entropy_with_logits(
        logits, one_hot_labels, name='xentropy')

    weight = tf.convert_to_tensor(weight,
                                  dtype=logits.dtype.base_dtype,
                                  name='loss_weight')
    loss = tf.multiply(weight, tf.reduce_mean(cross_entropy), name='value')
    tf.add_to_collection(LOSSES_COLLECTION, loss)
    return loss

def sigmoid_cross_entropy_loss(logits, one_hot_labels, label_smoothing=0,
                       weight=1.0, scope=None):
  """Define a Cross Entropy loss using softmax_cross_entropy_with_logits.

  It can scale the loss by weight factor, and smooth the labels.

  Args:
    logits: [batch_size, num_classes] logits outputs of the network .
    one_hot_labels: [batch_size, num_classes] target one_hot_encoded labels.
    label_smoothing: if greater than 0 then smooth the labels.
    weight: scale the loss by this factor.
    scope: Optional scope for name_scope.

  Returns:
    A tensor with the softmax_cross_entropy loss.
  """
  logits.get_shape().assert_is_compatible_with(one_hot_labels.get_shape())
  with tf.name_scope(scope, 'CrossEntropyLoss', [logits, one_hot_labels]):
    num_classes = one_hot_labels.get_shape()[-1].value
    one_hot_labels = tf.cast(one_hot_labels, logits.dtype)
    if label_smoothing > 0:
      smooth_positives = 1.0 - label_smoothing
      smooth_negatives = label_smoothing / num_classes
      one_hot_labels = one_hot_labels * smooth_positives + smooth_negatives
    cross_entropy = tf.contrib.nn.deprecated_flipped_sigmoid_cross_entropy_with_logits(
        logits, one_hot_labels, name='xentropy')

    weight = tf.convert_to_tensor(weight,
                                  dtype=logits.dtype.base_dtype,
                                  name='loss_weight')
    loss = tf.multiply(weight, tf.reduce_mean(cross_entropy), name='value')
    tf.add_to_collection(LOSSES_COLLECTION, loss)
    return loss

def weighted_sigmoid_cross_entropy_loss_backup(logits, one_hot_labels, label_weight, label_smoothing=0, weight=1.0, scope=None):
  """Define a Cross Entropy loss using softmax_cross_entropy_with_logits.

  It can scale the loss by weight factor, and smooth the labels.

  Args:
    logits: [batch_size, num_classes] logits outputs of the network .
    one_hot_labels: [batch_size, num_classes] target one_hot_encoded labels.
    label_smoothing: if greater than 0 then smooth the labels.
    weight: scale the loss by this factor.
    scope: Optional scope for name_scope.

  Returns:
    A tensor with the softmax_cross_entropy loss.
  """
  logits.get_shape().assert_is_compatible_with(one_hot_labels.get_shape())
  with tf.name_scope(scope, 'WeightedCrossEntropyLoss', [logits, one_hot_labels]):
    num_classes = one_hot_labels.get_shape()[-1].value
    one_hot_labels = tf.cast(one_hot_labels, logits.dtype)
    if label_smoothing > 0:
      smooth_positives = 1.0 - label_smoothing
      smooth_negatives = label_smoothing / num_classes
      one_hot_labels = one_hot_labels * smooth_positives + smooth_negatives
    
    #cross_entropy = tf.contrib.nn.deprecated_flipped_sigmoid_cross_entropy_with_logits(
    #    logits, one_hot_labels, name='xentropy')
    label_weight = tf.convert_to_tensor(label_weight, dtype=logits.dtype.base_dtype, name='attr_label_weight')
    lamda = (1-label_weight) * (1-one_hot_labels)
    
    cross_entropy = tf.add(lamda * logits + (label_weight * one_hot_labels + lamda) * tf.log(1 + tf.exp(-tf.abs(logits))) , tf.maximum(-logits, 0))

    weight = tf.convert_to_tensor(weight,
                                  dtype=logits.dtype.base_dtype,
                                  name='loss_weight')
    loss = tf.multiply(weight, tf.reduce_mean(cross_entropy), name='value')
    tf.add_to_collection(LOSSES_COLLECTION, loss)
    return loss
def weighted_sigmoid_cross_entropy_loss(logits, one_hot_labels, label_weight, label_smoothing=0, weight=1.0, scope=None):
  logits.get_shape().assert_is_compatible_with(one_hot_labels.get_shape())
  with tf.name_scope(scope, 'WeightedCrossEntropyLoss', [logits, one_hot_labels]):
    num_classes = one_hot_labels.get_shape()[-1].value
    one_hot_labels = tf.cast(one_hot_labels, logits.dtype)
    if label_smoothing > 0:
      smooth_positives = 1.0 - label_smoothing
      smooth_negatives = label_smoothing / num_classes
      one_hot_labels = one_hot_labels * smooth_positives + smooth_negatives
    label_weight = tf.convert_to_tensor(label_weight, dtype=logits.dtype.base_dtype, name='attr_label_weight')
    label_weight_w = []
    for i in range(one_hot_labels.get_shape()[0].value):
      label_weight_w.append(tf.exp(one_hot_labels[i] - tf.multiply( tf.sign(one_hot_labels[i]-0.5) ,label_weight)))
    label_weight_w = tf.convert_to_tensor(label_weight_w, dtype=logits.dtype.base_dtype, name='attr_label_weight')
    # label_weight_w = tf.exp(one_hot_labels[0] - tf.multiply( tf.sign(one_hot_labels[0]-0.5) ,label_weight))
    print(label_weight_w)

    cross_entropy = tf.contrib.nn.deprecated_flipped_sigmoid_cross_entropy_with_logits(
        logits, one_hot_labels, name='xentropy')
    weighted_cross_entropy = tf.multiply(cross_entropy, label_weight_w)

    weight = tf.convert_to_tensor(weight,
                                  dtype=logits.dtype.base_dtype,
                                  name='loss_weight')
    loss = tf.multiply(weight, tf.reduce_mean(weighted_cross_entropy), name='value')
    tf.add_to_collection(LOSSES_COLLECTION, loss)
    return loss
