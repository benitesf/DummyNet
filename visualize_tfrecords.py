# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 13:52:59 2018

@author: Edson BF

Usage: 
    # From tensorflow/
    # Visualize train or val records:
    #   set the filename of the TFRecord and maybe change the feature
    #   configuration in the same way files train and val were saved.
"""
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

filename = os.path.join('data', 'val.record')

def read_TFRF():    
    
    with tf.Session() as sess:
        feature = {'image': tf.FixedLenFeature((), tf.string, default_value=""),
                   'label': tf.FixedLenFeature((), tf.int64, default_value=0),
                   'filename': tf.FixedLenFeature((), tf.string, default_value="")}    
    
        # create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer([filename], num_epochs=1)
        
        # define a reader and read the next record
        reader = tf.TFRecordReader()    
        _, serialized_example = reader.read(filename_queue)
        
        # decode the record read by the reader    
        features = tf.parse_single_example(serialized_example, features=feature)
    
        # convert the image data from string back to the numbers    
        image = tf.decode_raw(features['image'], tf.float32)
        
        # cast label data into int32
        label = tf.cast(features['label'], tf.int32)
                        
        filena = tf.cast(features['filename'], tf.string)
        
        # reshape image data into the original shape
        image = tf.reshape(image, [32, 32, 3])

        # creates batches by randomly shuffling tensors        
        """images, labels, filenames = tf.train.shuffle_batch([image, label, filena], 
                batch_size=10,
                capacity=30, 
                num_threads=1, 
                min_after_dequeue=10)"""
    
        # initialize all gloabal and local variables
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        
        # create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        for batch_index in range(5):
            img, lbl, fil = sess.run([image, label, filena])
            img = img.astype(np.uint8)

            print('*'*40)
            print(img)
            print(lbl)
            print(fil)
            """for j in range(6):
                plt.subplot(2, 3, j + 1)
                plt.imshow(img)
                plt.title(fil)                        
                
            plt.show()"""
    
        # stop the threads
        coord.request_stop()
        
        # wait for threads to stop
        coord.join(threads)
        sess.close()
        
read_TFRF()