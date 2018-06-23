# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 08:57:04 2018

@author: Edson BF
"""

import tensorflow as tf
from os.path import join
from tensorflow.python.framework.ops import convert_to_tensor

class Dataset(object):

    """Create a Dataset object"""
    def __init__(self, filename_train=None, filename_val=None, batch_size=32):
        """Save parameters and run train or val operations"""
        self.FILENAME_TRAIN = filename_train
        self.FILENAME_VAL = filename_val        
        self.BATCH_SIZE = batch_size
        
    def train(self):
        return self._create_dataset(self.FILENAME_TRAIN)

    def validation(self):
        return self._create_dataset(self.FILENAME_VAL)
        
    def _create_dataset(self, filename):
        """Create a dataset and map records into tensors"""
        dataset = tf.data.TFRecordDataset(filename)
               
        # Map each record
        dataset = dataset.map(self._parse_function) 

        # shuffle dataset
        dataset = dataset.shuffle(buffer_size=1000)
       
        # batch the dataset
        dataset = dataset.batch(self.BATCH_SIZE)
        
        # repeat indefinitely
        dataset = dataset.repeat()
        
        return dataset
    
    def _parse_function(self, record):
        """Parse function for records"""
        # feature structure
        feature = {'image': tf.FixedLenFeature((), tf.string, default_value=""),
                   'label': tf.FixedLenFeature((), tf.int64, default_value=0),
                   'filename': tf.FixedLenFeature((), tf.string, default_value="")}
        
        # decode the record
        features = tf.parse_single_example(record, feature)
        
        # convert the image data from string back to the numbers
        # Info: the type depends on how the file was saved. (int32, float32, ...)
        image = tf.decode_raw(features['image'], tf.float32)
        
        # cast label data into int32
        label = tf.cast(features['label'], tf.int32)
        
        # cast filename data into string
        filename = tf.cast(features['filename'], tf.string)
        
        # reshape image data into the original shape
        # Info: Only for this experiment we reshape the image into [32, 32, 3],
        #   another dataset maybe needs a different reshape
        image = tf.reshape(image, [32, 32, 3])
        image /= 255
        
        return image, tf.one_hot(label, depth=10), filename
            

if __name__ == "__main__":
    
    filename_train = join('data', 'train.record')
    filename_val = join('data', 'val.record')
    
    dataset = Dataset(filename_train=filename_train, filename_val=filename_val)    
    
    train = dataset.train()
    val = dataset.validation()
    
    iterator = val.make_initializable_iterator()
    next_element = iterator.get_next()
    
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for i in range(5):
            try:
                img, lbl, fln = sess.run(next_element)
                print("#"*40)
                print(img[0,0:5,0:5,0])
                print(lbl)
                print(fln)
            except tf.errors.OutOfRangeError:
                print("End of dataset")
                break
    
        sess.close()