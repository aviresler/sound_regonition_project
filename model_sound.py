from __future__ import division
import tensorflow as tf
import sys
import time
import numpy as np
import os
from network_class import *
from sound_reader_kcross_val import SoundReaderKCrossValidation
from write_to_log import write_log
import csv
csvfile = "data1.csv"

class Model(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        
    # train
    def train(self):
        self.setup()

        self.sess.run(tf.global_variables_initializer())

        #Load the pre-trained model if provided
        if self.conf.pretrain_file is not '':
            self.load(self.loader, self.conf.pretrain_file)

        curr_valid_fold = self.conf.fold
        for epoch in range(self.conf.num_epochs + 1):
            start_time = time.time()
            train_offset = 0
            # training
            while train_offset < self.reader.get_train_length():
                if self.conf.is_between_class_train:
                    (batch_input, batch_labels) = self.reader.get_batch_bc(curr_valid_fold, train_offset, self.conf.batch_size)
                else:
                    (batch_input, batch_labels) = self.reader.get_batch(curr_valid_fold, train_offset, self.conf.batch_size)
                batch_input = batch_input.reshape((batch_input.shape[0], 1, batch_input.shape[1], 1))
                feed_dict = {self.net_input: batch_input, self.label_batch: batch_labels, self.keep_prob: 0.5,self.curr_step: epoch, self.isTrain: True}
                loss_value, _, pred, lr = self.sess.run([self.reduced_loss, self.train_optimizer, self.test_prediction, self.learning_rate], feed_dict=feed_dict)
               

                train_offset = train_offset + self.conf.batch_size

            # validation
            valid_offset = 0
            error_sum = 0
            while valid_offset < self.reader.get_valid_length():
                (valid_input, valid_labels) = self.reader.get_validation_batch_10_crops(curr_valid_fold, valid_offset, self.conf.valid_batch_size)
                valid_input = valid_input.reshape((valid_input.shape[0], 1, valid_input.shape[1], 1))
                feed_dict = {self.net_input: valid_input, self.label_batch: valid_labels, self.keep_prob: 1.0, self.isTrain: False}
                valid_pred = self.sess.run([self.test_prediction], feed_dict=feed_dict)
                valid_pred = np.squeeze(np.asarray(valid_pred))

                # averaging over 10 rows
                # averaging over 10 predictions to get the final predication of this sample
                valid_batch_pred_mat = np.zeros((self.conf.valid_batch_size, self.conf.num_classes))
                valid_batch_labels = np.zeros((self.conf.valid_batch_size, self.conf.num_classes))
                for mm in range(self.conf.valid_batch_size):
                    crop = valid_pred[mm*self.conf.num_of_valid_crop:(mm+1)*self.conf.num_of_valid_crop, :]
                    valid_batch_pred_mat[mm, :] = np.average(crop, axis=0)
                    valid_batch_labels[mm, :] = valid_labels[mm*self.conf.num_of_valid_crop, :]

                error_sum = error_sum + np.sum(np.argmax(valid_batch_pred_mat, 1) != np.argmax(valid_batch_labels, 1))
                valid_offset = valid_offset + self.conf.valid_batch_size

            valid_error = 100 * (error_sum/self.reader.get_valid_length())

            duration = time.time() - start_time
            epoch_str = 'epoch {:d} \t loss = {:.3f}, valid_err = {:.3f}, fold = {:d}, is_bc = {}, duration = {:.3f}, lr = {:.5f}'.format(epoch, loss_value, valid_error, curr_valid_fold, self.conf.is_between_class_train, duration,lr)
            print(epoch_str)
            write_log(epoch_str, 'fold' + str(curr_valid_fold) + '_bc_' + str(self.conf.is_between_class_train) + '_' + self.conf.logfile )
            # saving model of needed
            if epoch > 0:
                if epoch % self.conf.save_interval == 0:
                    self.save(self.saver, epoch)

    def setup(self):
        tf.set_random_seed(self.conf.random_seed)

        # Load reader
        with tf.name_scope("create_inputs"):
            self.reader = SoundReaderKCrossValidation(
            self.conf.data_dir,
            self.conf.data_list,
            self.conf.k_cross_val,
            self.conf.input_size,
            self.conf.num_classes,
            self.conf.num_of_valid_crop)
            self.net_input = tf.placeholder(tf.float32, shape=[None,1, self.conf.input_size, 1])
            self.keep_prob = tf.placeholder(tf.float32)
            self.label_batch = tf.placeholder(tf.float32, shape=[None, self.conf.num_classes])
            self.isTrain = tf.placeholder(tf.bool)
            self.epsilon_mat = tf.constant(1e-12, dtype=tf.float32,shape=[self.conf.batch_size, self.conf.num_classes])


        # create network
        net = DSRNet(self.net_input, self.conf.num_classes, self.isTrain,self.keep_prob)
        # Trainable Variables
        all_trainable = tf.trainable_variables()
        #todo - check if the regularization should be applied on all weights, including fully connected


        # Network raw output
        logits = net.outputs  # [batch_size, #calses]

        #if self.conf.is_between_class_train:
        #    soft_logits = tf.nn.softmax(logits)
        #    kl_matrix = self.label_batch * (tf.log(self.label_batch + self.epsilon_mat) - tf.log(soft_logits+self.epsilon_mat))
        #    loss = tf.reduce_sum(tf.reduce_sum(kl_matrix, 1))
        #else:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.label_batch))

        # L2 regularization
        l2_losses = [self.conf.weight_decay * tf.nn.l2_loss(v) for v in all_trainable if 'weights' in v.name]

        # Loss function
        self.reduced_loss = loss + tf.add_n(l2_losses)

        #Optimizer
        #learning rate configuration
        base_lr = tf.constant(self.conf.initial_learning_rate)
        self.curr_step = tf.placeholder(dtype=tf.float32, shape=())
        self.learning_rate = tf.train.piecewise_constant(self.curr_step, self.conf.lr_schedule, [base_lr, 0.1*base_lr,0.01*base_lr])
        self.learning_rate = tf.cond(self.curr_step < tf.constant(self.conf.warmup), lambda: 0.1*base_lr, lambda: base_lr )
        #Nestrov optimizer with momentum
        optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.conf.momentum, use_nesterov= True)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for collecting moving_mean and moving_variance of bach normailzaition
        with tf.control_dependencies(update_ops):
            self.train_optimizer = optimizer.minimize(self.reduced_loss)

        # Saver for storing checkpoints of the model
        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep= 1)

        # Loader for loading the pre-trained model
        self.loader = tf.train.Saver(var_list=tf.global_variables())

        #Predictions for the trainings
        self.test_prediction = tf.nn.softmax(logits)

        #validation!!
        #valid_net = DSRNet(self.valid_input, self.conf.num_classes, False, 1,True)
        #self.valid_predictions = tf.nn.softmax(valid_net.outputs)

    def save(self, saver, step):
        '''
        Save weights.
        '''
        model_name = 'model.ckpt'
        checkpoint_path = os.path.join(self.conf.modeldir + '_' + str(self.conf.fold) + '_bc_' + str(self.conf.is_between_class_train), model_name)
        if not os.path.exists(self.conf.modeldir):
            os.makedirs(self.conf.modeldir)
        saver.save(self.sess, checkpoint_path, global_step=step)
        print('The checkpoint has been created.')


    def load(self, saver, filename):
        '''
        Load trained weights.
        ''' 
        saver.restore(self.sess, filename)
        print("Restored model parameters from {}".format(filename))

    def predictions_error(self,predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) != np.argmax(labels,1))/ predictions.shape[0])
    
    
    
    
    
    
    
    
    
    
    
