
import tensorflow as tf
import numpy as np
import six

class DSRNet(object):
  
    def __init__(self, inputs, num_classes, phase,keep_prob):
        self.inputs = inputs
        self.keep_prob = keep_prob
        self.num_classes = num_classes
        self.channel_axis = 3
        self.phase = phase # train (True) or test (False), for BN layers in the decoder
        self.build_network()
        
    def build_network(self):
        print("building DSRNet: ")
        print("convolution blocks")
        conv1 = self._conv2d(self.inputs,[1,64],32,[1,2], name='conv1')
        bn_conv1 = self._batch_norm(conv1, name='bn_conv1', is_training=self.phase, activation_fn=tf.nn.relu)
        conv2 = self._conv2d(bn_conv1,[1,16],64,[1,2], name='conv2')
        bn_conv2 = self._batch_norm(conv2, name='bn_conv2', is_training=self.phase, activation_fn=tf.nn.relu)
        pool2 = self._max_pool2d(bn_conv2, [1,64],[1,64], name='pool2')
        print("swap axes")
        swapped_pool2 = tf.transpose(pool2, [0,3, 2, 1])
        conv3 = self._conv2d(swapped_pool2,[8,8],32,[1,1], name='conv3')
        bn_conv3 = self._batch_norm(conv3, name='bn_conv3', is_training=self.phase, activation_fn=tf.nn.relu)
        conv4 = self._conv2d(bn_conv3,[8,8],32,[1,1], name='conv4')
        bn_conv4 = self._batch_norm(conv4, name='bn_conv4', is_training=self.phase, activation_fn=tf.nn.relu)
        pool4 = self._max_pool2d(bn_conv4, [5,3],[5,3], name='pool4')
        conv5 = self._conv2d(pool4,[1,4],64,[1,1], name='conv5')
        bn_conv5 = self._batch_norm(conv5, name='bn_conv5', is_training=self.phase, activation_fn=tf.nn.relu)    
        conv6 = self._conv2d(bn_conv5,[1,4],64,[1,1], name='conv6')
        bn_conv6 = self._batch_norm(conv6, name='bn_conv6', is_training=self.phase, activation_fn=tf.nn.relu)
        pool6 = self._max_pool2d(bn_conv6, [1,2],[1,2], name='pool6')
        conv7 = self._conv2d(pool6,[1,2],128,[1,1], name='conv7')
        bn_conv7 = self._batch_norm(conv7, name='bn_conv7', is_training=self.phase, activation_fn=tf.nn.relu)        
        conv8 = self._conv2d(bn_conv7,[1,2],128,[1,1], name='conv8')
        bn_conv8 = self._batch_norm(conv8, name='bn_conv8', is_training=self.phase, activation_fn=tf.nn.relu) 
        pool8 = self._max_pool2d(bn_conv8, [1,2],[1,2], name='pool8')
        conv9 = self._conv2d(pool8,[1,2],256,[1,1], name='conv9')
        bn_conv9 = self._batch_norm(conv9, name='bn_conv9', is_training=self.phase, activation_fn=tf.nn.relu)        
        conv10 = self._conv2d(bn_conv9,[1,2],256,[1,1], name='conv10')
        bn_conv10 = self._batch_norm(conv10, name='bn_conv10', is_training=self.phase, activation_fn=tf.nn.relu) 
        pool10 = self._max_pool2d(bn_conv10, [1,2],[1,2], name='pool10')
        print("fully connected blocks")  
        shape = pool10.get_shape().as_list()
        #tf.shape(pool10)[0]
        #reshape10 = tf.reshape(pool10, [[0], shape[1] * shape[2] * shape[3]])
        reshape10 = tf.reshape(pool10, [tf.shape(pool10)[0], tf.shape(pool10)[1] * tf.shape(pool10)[2] * tf.shape(pool10)[3]])
        fc11 = self._fc(reshape10, shape[1] * shape[2] * shape[3], 4096, name= 'fc11')
        #fc11 = self._fc(reshape10, tf.shape(pool10)[1] * tf.shape(pool10)[2] * tf.shape(pool10)[3], 4096, name='fc11')
        #fc11 = self._fc(reshape10, None, 4096, name='fc11')
        dropped_fc11 = tf.nn.dropout(fc11, self.keep_prob)
        fc12 = self._fc(dropped_fc11, 4096, 4096, name= 'fc12')
        dropped_fc12 = tf.nn.dropout(fc12, self.keep_prob)
        self.outputs = self._fc(dropped_fc12, 4096, self.num_classes, name= 'fc13')
        
   
        
    def _conv2d(self, x, kernel_size, num_o, stride, name, biased=False):
        """
        Conv2d without BN or relu.
        """
        num_x = x.shape[self.channel_axis].value
        with tf.variable_scope(name) as scope:
            w = tf.get_variable('weights', shape=[kernel_size[0], kernel_size[1], num_x, num_o])
            s = [1, stride[0], stride[1], 1]
            o = tf.nn.conv2d(x, w, s, padding='VALID')
            if biased:
                b = tf.get_variable('biases', shape=[num_o])
                o = tf.nn.bias_add(o, b)
        return o
        
    def _batch_norm(self, x, name, is_training, activation_fn, trainable=True):
		# For a small batch size, it is better to keep 
		# the statistics of the BN layers (running means and variances) frozen, 
		# and to not update the values provided by the pre-trained model by setting is_training=False.
		# Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
		# if they are presented in var_list of the optimiser definition.
		# Set trainable = False to remove them from trainable_variables.
        with tf.variable_scope(name+'/BatchNorm') as scope:
            o = tf.contrib.layers.batch_norm(
				x,
				scale=True,
				activation_fn=activation_fn,
				is_training=is_training,
				trainable=trainable,
				scope=scope)
        return o
    
    def _max_pool2d(self, x, kernel_size, stride, name):
        k = [1, kernel_size[0], kernel_size[1], 1]
        s = [1, stride[0], stride[1], 1]
        return tf.nn.max_pool(x, k, s, padding='VALID', name=name)
    
    def _fc(self, x, num_i, num_o, name):
        """
        fully connected
        """
        with tf.variable_scope(name) as scope:
            w = tf.get_variable('weights', shape=[num_i, num_o])
            b = tf.get_variable('biases', shape=[num_o])
            o = tf.matmul(x, w) + b
        return o   
        