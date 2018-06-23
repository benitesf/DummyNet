# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 09:53:17 2018

@author: Edson BF
"""
import tensorflow as tf
import numpy as np
from model import DummyNet

from math import floor
from dataset import Dataset

"""
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '', 'Directory to save the checkpoints and training summaries.')
tf.app.flags.DEFINE_string('train_file', 'data/train.record', 'TFRecord file to train the network.')
tf.app.flags.DEFINE_integer('max_steps', 1000000, 'Number of batches to run')
tf.app.flags.DEFINE_integer('log_frequency', 10, 'How often to log results to the console')
"""


# Hyper-parameters
learning_rate = 1E-2
batch_size = 128
epochs = 10
display_step = 50  # save summary after a number of steps
display_epoch = 5  # save model after a number of epochs

# Dataset parameters
classes = 10
training_size = 50000
train_file = 'data/train.record'

summary_addrs = 'summary'
ckpt_addrs = 'ckpt'


def main():
    """Train the network which is imported from model"""
    with tf.Graph().as_default():
        # Get the global step to track training process
        global_step = tf.train.get_or_create_global_step()
        
        # Create a placeholder to switch on/off BN
        is_training = tf.placeholder(tf.bool, name='is_training')
        
        # Create a placeholder
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        
        with tf.device('/cpu:0'):
            # Get dataset object
            ds = Dataset(filename_train=train_file, batch_size=batch_size)
        
            # Get train dataset
            train_ds = ds.train()                    
            
            # Create the iterator
            iterator = train_ds.make_one_shot_iterator()
            
            # Get features and labels
            x, y, _ = iterator.get_next()
                                            
        # Build the model
        my_net = DummyNet(x, classes=classes, keep_prob=keep_prob, is_training=is_training)
        
        # Inference op
        #infe = my_net.inference()
        
        # Loss op
        loss = my_net.loss(y)
        
        # Accuracy op
        accuracy = my_net.accuracy(y)
        
        # Train op
        with tf.name_scope('train'): 
            # Create optimizer and apply gradient descent to the trainable variables
            train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # Merge all summaries into single op    
        merged_summary_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter(summary_addrs)
        
        # Initialize an saver for store model checkpoints
        saver = tf.train.Saver(max_to_keep=5)
        
        # Create session        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            # Add the model graph to TensorBoard
            writer.add_graph(sess.graph)
            
            print("Training network...")
            
            step_per_epoch = int(floor(training_size / batch_size))
            
            # Loop over number of epochs
            import time
            
            start = time.time()
            for epoch in range(epochs):
                epoch_info = "[{}/{}]".format(epoch + 1, epochs)                                

                all_loss = []       
                
                # Training process
                for _ in range(step_per_epoch):
                    _, curr_loss = sess.run([train_op, loss], feed_dict={keep_prob: 0.5, is_training: True})    
                    all_loss.append(curr_loss)
                    
                    # Generate summary with the current batch of data and write to file
                    step = tf.train.global_step(sess, global_step)
                    
                    if step % display_step == 0:                        
                        summary = sess.run(merged_summary_op, feed_dict={keep_prob: 1.0, is_training: False})
                        writer.add_summary(summary, step)
                                                                                                    
                print('{} loss: {:.4f}'.format(epoch_info, np.mean(all_loss)))
                
                # save checkpoint of the model every 5 epochs
                if epoch % display_epoch == 0:
                    saver.save(sess, ckpt_addrs, global_step=step)
            end = time.time()
            print(end - start)

if __name__ == '__main__':    
    main()
