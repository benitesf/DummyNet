# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 10:12:11 2018

@author: Edson BF

Usage:
    # From tensorflow/
    # Create train TFRecord
    python generate_tfrecord.py --data_dir=data/cifar-10-batches-py --set=train --output_path=data --filename=train.record
    
    # Create val TFRecord
    python generate_tfrecord.py --data_dir=data/cifar-10-batches-py --set=val --output_path=data --filename=val.record
    
Info:
    This script create and save tfrecord files from a batch files structure that
    provide cifar-10 challenge. Consider to make some modifications for other
    type of structure.
    
    https://www.cs.toronto.edu/~kriz/cifar.html
"""
import os
import pickle
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '', 'Directory to read images.')
tf.app.flags.DEFINE_string('filename', '', 'Filename of the tf record.')
tf.app.flags.DEFINE_string('set', '', 'Set type, train or eval')
tf.app.flags.DEFINE_string('output_path', '', 'Path to save TFRecord.')


def unpickle(file):
    """Unpickle images"""    
    with open(file, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    return d


def visualize_images(images):
    import matplotlib.pyplot as plt
    fig, axes1 = plt.subplots(5, 5, figsize=(5,5))
    for j in range(5):
        for k in range(5):
            i = np.random.choice(range(len(images)))
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(images[i : i + 1][0])


def unpack_images(images):
    """Reshape from flatten to RGB images"""
    batch = images.shape[0]    
    return images.reshape(batch, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)


"""Conver data to features"""
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_record(data_list, filepath):
    """Create a TFRecord File and save it."""
    # Open the writer
    writer = tf.python_io.TFRecordWriter(filepath)
    
    print('Saving data in ' + filepath)
    
    for d in data_list:
        # Get all items from dict
        _, labels, data, filenames = d.items()
        
        # unpack images from data
        images = unpack_images(data[1])
        
        # Iterate over all the images
        for img, lbl, fname in zip(images, labels[1], filenames[1]):
            # Create feature
            feat = {'image': _bytes_feature(tf.compat.as_bytes(img.tostring())),
                    'label': _int64_feature(lbl),
                    'filename': _bytes_feature(fname)}
            
            # Create and example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feat))
            
            # Serialize to string and write on file
            writer.write(example.SerializeToString())
            
    # Close
    writer.close()
        
        
def main(_):
    if FLAGS.set == 'train':
        # Read all the batches [1-5]
        filename = os.path.join(FLAGS.data_dir, 'data_batch_')
        data = [unpickle(filename + str(i)) for i in range(1, 6)]
    else:
        # Read test batch
        filename = os.path.join(FLAGS.data_dir, 'test_batch')
        data = [unpickle(filename)]
    
    filepath = os.path.join(FLAGS.output_path, FLAGS.filename)
    create_record(data, filepath)
    
    

if __name__ == '__main__':
    tf.app.run()