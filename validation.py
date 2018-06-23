# -*- coding: utf-8 -*-
"""
Created on Fri May  4 11:28:46 2018

@author: Edson BF
"""

import tensorflow as tf
import numpy as np

from os.path import join
from math import floor
from dataset import Dataset

# Reset the current graph
tf.reset_default_graph()

# Checkpoint address
ckpt_dir = 'ckpt'

summary_addrs = 'summary/validation/testing'

# Hyperparameters
batch_size = 32

# Dataset parameters
classes = 10
validation_size = 10000
val_file = 'data/val.record'
    
with tf.device('/cpu:0'):
    # Get dataset object
    ds = Dataset(filename_val=val_file, batch_size=batch_size)

    # Get validation dataset
    val_ds = ds.validation()
    
    # Create the iterator
    iterator = val_ds.make_one_shot_iterator()
    
    # Get features and labels
    x, y, _ = iterator.get_next()        

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
        
    # Read the checkpoint and restore it
    latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    saver = tf.train.import_meta_graph(latest_ckpt + '.meta')    
    saver.restore(sess, latest_ckpt)
    
    # Get the tensors from the graph
    g = tf.get_default_graph()
    prediction = g.get_tensor_by_name('Prediction/prediction:0')
    accuracy = g.get_tensor_by_name('Accuracy/accuracy:0')
    is_training = g.get_tensor_by_name('is_training:0')
    keep_prob = g.get_tensor_by_name('keep_prob:0')
            
    # Number of steps
    step_per_epoch = floor(validation_size / batch_size)
    
    print('Validating dataset...')

    score = []

    for _ in range(step_per_epoch):
        score.append(sess.run([accuracy], feed_dict={keep_prob: 1.0, is_training: True}))
    
    step = latest_ckpt.split('-')[-1]
    score = np.mean(score)
    
    print('Step[{}] , acc[{:.4f}]'.format(step, score))
    
    with tf.name_scope('Metrics/'):
        summary_op = tf.summary.scalar('accuracy_validation', score)
    
    writer = tf.summary.FileWriter(summary_addrs)
    summary = sess.run(summary_op)
    writer.add_summary(summary, step)