# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 11:18:18 2018

@author: Edson BF
"""

import tensorflow as tf


def _variable_summaries(var):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('summary', var)

def _loss_to_summary(loss):
    """Add loss measures to summary.
    
    Add summary for 'Loss' and 'Loss/avg'.
    
    Parameters
    ----------
    loss: from loss op.
    
    Returns
    -------
    nothing
    """
    #tf.add_to_collection(tf.GraphKeys.LOSSES, loss)    
    #total_loss = tf.add_n(tf.get_collection('losses'))        
    #avg_loss = tf.reduce_mean(tf.get_collection_ref(tf.GraphKeys.LOSSES)) 
    #tf.losses.add_loss(loss)
    # Add the loss to summary    
    #tf.summary.scalar('loss', loss)
    #tf.summary.scalar('total_loss', total_loss)
    #tf.summary.scalar('avg_loss', avg_loss)


def _accuracy_to_summary(acc):
    """Add accuracy measures to summary.    
    """
    tf.add_to_collection('accuracies', acc)
    avg_acc = tf.reduce_mean(tf.get_collection('accuracies'))
    
    tf.summary.scalar('accuracy', acc)
    tf.summary.scalar('avg_accuracy', avg_acc)


class DummyNet(object):
    """Implementation of a DummyNet class"""    
    
    def __init__(self, inputs, classes, keep_prob, is_training):
        """Create the computational graph of dummynet
        
        Parameters
        ----------
        inputs
            Input images shapes
        classes
            Number of classes of the problem
            
        Returns
        -------
        nothing
        """
        self.INPUTS = inputs
        self.CLASSES = classes
        self.KEEP_PROB = keep_prob
        self.IS_TRAINING = is_training
        self._create()
        
    def _create(self):
        """Create the graph"""
        with tf.variable_scope('conv-module-1'):
            # Conv1-1 (w ReLU) -> Conv1-2 (w ReLU) -> Pool
            conv1_1 = self._conv(self.INPUTS, 64, [3, 3], 1, name='conv1')
            conv1_2 = self._conv(conv1_1, 64, [3, 3], 1, name='conv2')        
            pool1 = self._pool(conv1_2, [2, 2], 2, name='pool')
        
        with tf.variable_scope('conv-module-2'):
            # Conv2-1 (w ReLU) -> Conv2-2 (w ReLU) -> Pool
            conv2_1 = self._conv(pool1, 128, [3, 3], 1, name='conv1')
            conv2_2 = self._conv(conv2_1, 128, [3, 3], 1, name='conv2')
            pool2 = self._pool(conv2_2, [2, 2], 2, name='pool')
        
        with tf.variable_scope('fc-module'):
            # Flatten -> FC3-1 (w ReLU) Dropout -> FC3-2 (w ReLU) Dropout
            flatten = tf.layers.flatten(pool2)
            fc3_1 = tf.layers.dense(flatten, 2048, activation=tf.nn.relu, name='fc1')
            dropout_fc3_1 = tf.nn.dropout(fc3_1, self.KEEP_PROB, name='dropout-fc1') 
            fc3_2 = tf.layers.dense(dropout_fc3_1, 2048, activation=tf.nn.relu, name='fc2')
            dropout_fc3_2 = tf.nn.dropout(fc3_2, self.KEEP_PROB, name='dropout-fc2')
        
            # Save summary
            tf.summary.histogram('fc1/activations', fc3_1)
            tf.summary.histogram('fc2/activations', fc3_2)
            
        with tf.variable_scope('output-module'):
            self.output_layer = tf.layers.dense(dropout_fc3_2, self.CLASSES, activation=tf.nn.softmax, name='output')
            
            # Save summary
            tf.summary.histogram('activations', self.output_layer)
        
    
    def _conv(self, inputs, num_filters, filter_size, stride, name, padding='SAME'):
        """Create a convolution layer with ReLU activation"""      
        with tf.variable_scope(name):
            input_channel = int(inputs.get_shape().as_list()[-1])
            shape = [filter_size[0], filter_size[1], input_channel, num_filters]
            
            # Create tf variables for the weights and biases for the conv layer            
            with tf.variable_scope('weights') as scope:
                weights = tf.get_variable(scope.name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
                _variable_summaries(weights)
            
            with tf.variable_scope('biases') as scope:
                biases = tf.get_variable(scope.name, shape=[num_filters], initializer=tf.contrib.layers.xavier_initializer()) 
                _variable_summaries(biases)
            
            # Convolve inputs with weights
            convolve = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding=padding)
            
            # Batch Normalization
            normalized = self._batch_norm(convolve)
            
            # Add biases
            pre_activation = tf.nn.bias_add(normalized, biases)            
            
            # Apply ReLU function
            activation = tf.nn.relu(pre_activation)
            
            # Save summary
            tf.summary.histogram('activations', activation)
            
            return activation
        
    def _batch_norm(self, x):
        """Create a batch normalization transform.
        
        Parameters
        ----------
        x: Tensor, 4D BHWD input maps
        is_training: boolean, true indicates is training process
        
        Returns
        -------
        normed: batch-normalized maps
        """
        with tf.variable_scope('batch-norm'):
            depth = int(x.get_shape().as_list()[-1])
            shift = tf.Variable(tf.constant(0.0, shape=[depth]), name='shift', trainable=True)
            scale = tf.Variable(tf.constant(1.0, shape=[depth]), name='scale', trainable=True)
            
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.9)
            
            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)
            
            mean, var = tf.cond(self.IS_TRAINING, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))                        
            normed = tf.nn.batch_normalization(x, mean, var, shift, scale, 1e-3)            
        return normed
    
    def _pool(self, inputs, pool_size, stride, name, padding='SAME'):
        """Create a max-pooling layer"""
        # prepare args
        ksize = [1, pool_size[0], pool_size[1], 1]
        strides = [1, stride, stride, 1]
        
        return tf.nn.max_pool(inputs, ksize=ksize, strides=strides, padding=padding, name=name)
    
    def inference(self):
        """Do the inference of the net"""
        with tf.variable_scope('Inference'):
            return self.output_layer
        
    def prediction(self):
        """Calculate the prediction.
        
        Returns
        -------
        return class prediction (e.g. [0,0,1,5,7, .., 6])
        """        
        with tf.name_scope('Prediction'):
            prediction = tf.argmax(self.inference(), axis=1, name='prediction')
            
        return prediction
    
    def _L2loss(self):
        with tf.name_scope('L2Loss'):
            trainable = tf.trainable_variables()
            return tf.add_n([tf.nn.l2_loss(v) for v in trainable if 'bias' not in v.name]) * 0.001
    
    def loss(self, labels):
        """Calculate the network loss.
        
        Add summary for "Loss".
        
        Parameters
        ----------
        logits: From inference()
        labels: Labels of the examples
            
        """
        with tf.name_scope('Loss'):
            #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.inference(), labels=labels))
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.inference(), labels=labels)) + self._L2loss()
        
        with tf.name_scope('Metrics/'):
            tf.summary.scalar('loss', loss)
        return loss
    
    def accuracy(self, labels):
        """Calculate the accuracy of the network.
        
        Parameters
        ----------
        labels: Labels of the examples.
        """
        comparison = tf.equal(self.prediction(), tf.argmax(labels, 1))
        with tf.name_scope('Accuracy'):            
            acc = tf.reduce_mean(tf.cast(comparison, tf.float32), name='accuracy')
        
        with tf.name_scope('Metrics/'):
            tf.summary.scalar('accuracy', acc)
        return acc
        
        
#tf.reset_default_graph()
#x = tf.placeholder(tf.float32, [None, 32, 32, 3])
#mynet = DummyNet(x, 10)