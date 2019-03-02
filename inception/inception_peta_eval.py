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
"""A library to evaluate Inception on a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import os.path
import time


import numpy as np
import tensorflow as tf

from inception import image_processing
from inception import inception_peta_model as inception

num_classes = 35

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/home/sangliufang/tensorflow/models/research/inception/inception/rap_train',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/imagenet_train',
                           """Directory where to read model checkpoints.""")

# Flags governing the frequency of the eval.
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

tf.app.flags.DEFINE_boolean('pad', False,
                            """Whether to run eval only once.""")

# Flags governing the data used for the eval.
tf.app.flags.DEFINE_integer('num_examples', 8317,
                            """Number of examples to run. Note that the eval """
                            """ImageNet dataset contains 50000 examples.""")
tf.app.flags.DEFINE_string('subset', 'validation',
                           """Either 'validation' or 'train'.""")

def mA(attr, gt):
    num = attr.__len__()
    num_attr = attr[0].__len__()
    # for i in range(num):
    #   if(i % 100 == 0):
    #     print(attr[i])

    challenging = []
    acc_collect = []
    accuracy_neg = []
    accuracy_pos = []
    accuracy_all = []
    for i in range(num_attr):
        print('--------------------------------------------')
        print(i)
        print(sum([attr[j][i] for j in range(num)]), \
                ':', sum([attr[j][i] * gt[j][i] for j in range(num)]), \
                ':', sum([gt[j][i] for j in range(num)]))
        # print(sum([attr[j][i] * gt[j][i] for j in range(num)]) / sum([gt[j][i] for j in range(num)]))
        accuracy_pos.append(sum([attr[j][i] * gt[j][i] for j in range(num)]) / sum([gt[j][i] for j in range(num)]))

        print("accuracy_pos: {}\n".format(accuracy_pos[i]))
        print(sum([(1 - attr[j][i]) for j in range(num)]), \
                ':', sum([(1 - attr[j][i]) * (1 - gt[j][i]) for j in range(num)]), \
                ':', sum([(1 - gt[j][i]) for j in range(num)]))
        # print(sum([(1 - attr[j][i]) * (1 - gt[j][i]) for j in range(num)]) / sum([(1 - gt[j][i]) for j in range(num)]))
        # accuracy_pos.append(sum([attr[j][i] * gt[j][i] for j in range(num)]) / sum([gt[j][i] for j in range(num)]))

        # print("accuracy_pos: {}\n".format(accuracy_pos[i]))
        accuracy_neg.append(sum([(1 - attr[j][i]) * (1 - gt[j][i]) for j in range(num)]) / sum([(1 - gt[j][i]) for j in range(num)]))

        print("accuracy_neg: {}\n".format(accuracy_neg[i]))

        accuracy_all.append((accuracy_neg[i]+accuracy_pos[i])/2)
        print("accuracy_all: {}\n".format(accuracy_all[i]))

        # acc = (sum([attr[j][i] * gt[j][i] for j in range(num)]) / sum([gt[j][i] for j in range(num)]) + sum(
        #         [(1 - attr[j][i]) * (1 - gt[j][i]) for j in range(num)]) / sum([(1 - gt[j][i]) for j in range(num)])) / 2
        # acc_collect.append(acc)
        # print("accuracy: {}\n".format(acc))
        if accuracy_all[i] < 0.75:
            challenging.append(i)

        # mA = (sum([(
        #         sum([attr[j][i] * gt[j][i] for j in range(num)])
        #            / sum([gt[j][i] for j in range(num)])
        #            + sum([(1 - attr[j][i]) * (1 - gt[j][i]) for j in range(num)])
        #            / sum([(1 - gt[j][i]) for j in range(num)])
        #    ) for i in range(num_attr)])) / (2 * num_attr)
    return np.mean(accuracy_all), accuracy_all, challenging

def example_based(attr, gt):
  num = attr.__len__()
  num_attr = attr[0].__len__()

  acc = 0
  prec = 0
  rec = 0
  f1 = 0

  attr = np.array(attr).astype(bool)
  gt = np.array(gt).astype(bool)
  
  for i in range(num):
    intersect = sum((attr[i] & gt[i]).astype(float))
    union = sum((attr[i] | gt[i]).astype(float))
    attr_sum = sum((attr[i]).astype(float))
    gt_sum = sum((gt[i]).astype(float))
    
    acc += 0 if union == 0  else  intersect / union
    prec += 0 if attr_sum == 0 else intersect / attr_sum
    rec += 0 if  gt_sum == 0 else  intersect / gt_sum
  
  acc /= num
  prec /= num
  rec /= num
  f1 = 2 * prec * rec / (prec + rec)

  return acc, prec, rec, f1

def softmax_or_not(predict, is_softmax=False, threshold=[0.5]*num_classes):
  res = []

  if not is_softmax:
    for i in range(FLAGS.batch_size):
        res1 = [0] * num_classes
        for j in range(num_classes):
          if predict[i][j] >= threshold[j]:
            res1[j] = 1
        res.append(res1)
    return res

  for i in range(FLAGS.batch_size):
    res1 = [0] * num_classes
    for j in range(num_classes):
      if j == 1:
        max_id, max_v = -1, -1
        for k in range(1, 4):
          if predict[i][k] > max_v:
            max_id = k
            max_v = predict[i][k]
        res1[max_id] = 1
      elif j == 4:
        max_id, max_v = -1, -1
        for k in range(4, 7):
          if predict[i][k] > max_v:
            max_id = k
            max_v = predict[i][k]
        if max_v >= threshold[j]:
          res1[max_id] = 1
      elif j == 7:
        max_id, max_v = -1, -1
        for k in range(7, 9):
          if predict[i][k] > max_v:
            max_id = k
            max_v = predict[i][k]
        if max_v >= threshold[j]:
          res1[max_id] = 1
      elif j == 9:
        max_id, max_v = -1, -1
        for k in range(9, 11):
          if predict[i][k] > max_v:
            max_id = k
            max_v = predict[i][k]
        if max_v >= threshold[j]:
          res1[max_id] = 1
      elif j < 1  or j >= 11:
        if predict[i][j] >= threshold[j]:
          res1[j] = 1
      else:
        continue
    res.append(res1)
  return res

def _eval_once_mA(saver, mA_op, summary_writer, summary_op):
  with tf.Session() as sess:
    if os.path.exists(FLAGS.checkpoint_dir):
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        if os.path.isabs(ckpt.model_checkpoint_path):
          # Restores from checkpoint with absolute path.
          saver.restore(sess, ckpt.model_checkpoint_path)
        else:
          # Restores from checkpoint with relative path.
          saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
                                           ckpt.model_checkpoint_path))

        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/imagenet_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('Successfully loaded model from %s at step=%s.' %
              (ckpt.model_checkpoint_path, global_step))
      else:
        print('No checkpoint file found')
        return
    else:
        saver.restore(sess, FLAGS.checkpoint_dir)
        global_step = FLAGS.checkpoint_dir.split('/')[-1].split('-')[-1]
        print('Successfully loaded model from %s at step=%s.' %
              (FLAGS.checkpoint_dir, global_step))


    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      attrs = np.zeros((total_sample_count , num_classes))
      labels = np.zeros((total_sample_count, num_classes))
      step = 0

      print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
      start_time = time.time()
      while step < num_iter and not coord.should_stop():
        predict, label = sess.run(mA_op)
     
        threshold = [0.0] * num_classes
        predict = softmax_or_not(predict, is_softmax=False, threshold=threshold)
        attrs[step*FLAGS.batch_size : (step+1)*FLAGS.batch_size,:] = predict
        labels[step*FLAGS.batch_size : (step+1)*FLAGS.batch_size,:] = label
        # np.append(attrs, predict, axis=0)
        # np.append(labels, label, axis=0)
          
        step += 1
        if step % 20 == 0:
          duration = time.time() - start_time
          sec_per_batch = duration / 20.0
          examples_per_sec = FLAGS.batch_size / sec_per_batch
          print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                'sec/batch)' % (datetime.now(), step, num_iter,
                                examples_per_sec, sec_per_batch))
          start_time = time.time()

      # Compute mA.
      print(mA(attrs, labels))
      print(example_based(attrs, labels))
      #mA = sum_mA / total_sample_count
      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

def _eval_once(saver, summary_writer, top_1_op, top_5_op, summary_op):
  """Runs Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_1_op: Top 1 op.
    top_5_op: Top 5 op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    if os.path.exists(FLAGS.checkpoint_dir):
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        if os.path.isabs(ckpt.model_checkpoint_path):
          # Restores from checkpoint with absolute path.
          saver.restore(sess, ckpt.model_checkpoint_path)
        else:
          # Restores from checkpoint with relative path.
          saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
                                           ckpt.model_checkpoint_path))

        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/imagenet_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('Successfully loaded model from %s at step=%s.' %
              (ckpt.model_checkpoint_path, global_step))
      else:
        print('No checkpoint file found')
        return
    else:
        saver.restore(sess, FLAGS.checkpoint_dir)
        global_step = FLAGS.checkpoint_dir.split('/')[-1].split('-')[-1]
        print('Successfully loaded model from %s at step=%s.' %
              (FLAGS.checkpoint_dir, global_step))

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      # Counts the number of correct predictions.
      count_top_1 = 0.0
      count_top_5 = 0.0
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0

      print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
      start_time = time.time()
      while step < num_iter and not coord.should_stop():
        top_1, top_5 = sess.run([top_1_op, top_5_op])
        count_top_1 += np.sum(top_1)
        count_top_5 += np.sum(top_5)
        step += 1
        if step % 20 == 0:
          duration = time.time() - start_time
          sec_per_batch = duration / 20.0
          examples_per_sec = FLAGS.batch_size / sec_per_batch
          print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                'sec/batch)' % (datetime.now(), step, num_iter,
                                examples_per_sec, sec_per_batch))
          start_time = time.time()

      # Compute precision @ 1.
      precision_at_1 = count_top_1 / total_sample_count
      recall_at_5 = count_top_5 / total_sample_count
      print('%s: precision @ 1 = %.4f recall @ 5 = %.4f [%d examples]' %
            (datetime.now(), precision_at_1, recall_at_5, total_sample_count))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision_at_1)
      summary.value.add(tag='Recall @ 5', simple_value=recall_at_5)
      summary_writer.add_summary(summary, global_step)

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

def mean_accuracy(logits, labels):
    print(logits)
    labels = tf.to_float(labels)
    # logits = tf.sigmoid(logits)
    # logits = (tf.sign(tf.sigmoid(logits) - 0.5) + 1) /2



    #num_pp = tf.reduce_sum(predict * labels, 0)
    #num_nn = tf.reduce_sum((predict-1) * (labels - 1), 0)
    #num_pn = tf.reduce_sum(predict * (1 - labels), 0)
    #num_np = tf.reduce_sum((1-predict) * labels, 0)
    return logits, labels

def evaluate(dataset):
  """Evaluate model on Dataset for a number of steps."""
  with tf.Graph().as_default():
    # Get images and labels from the dataset.
    images, labels, rois = image_processing.inputs(dataset)

    # Number of classes in the Dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class.
    num_classes = dataset.num_classes()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits, _ = inception.inference_roi_lstm_bn(images, rois, num_classes)

    # Calculate predictions.
    #top_1_op = tf.nn.in_top_k(logits, labels, 1)
    #top_5_op = tf.nn.in_top_k(logits, labels, 5)
    mA_op = mean_accuracy(logits, labels)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        inception.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    graph_def = tf.get_default_graph().as_graph_def()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir,
                                            graph_def=graph_def)

    while True:
      #_eval_once(saver, summary_writer, top_1_op, top_5_op, summary_op)
      _eval_once_mA(saver, mA_op, summary_writer, summary_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)
