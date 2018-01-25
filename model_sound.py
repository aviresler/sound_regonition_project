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

        if False == self.conf.is_test_10_crops:
            for epoch in range(self.conf.num_epochs + 1):
                epoc_error_rate_list = []
                start_time = time.time()
                #k-cross validation
                for fold_index in range(self.conf.k_cross_val):
                    train_offset = 0
                    #runnning on the entire train set
                    while train_offset < self.reader.get_train_length():
                        if self.conf.is_between_class_train:
                            (batch_input, batch_labels) = self.reader.get_batch_bc(fold_index, train_offset, self.conf.batch_size)
                        else:
                            (batch_input, batch_labels) = self.reader.get_batch(fold_index, train_offset, self.conf.batch_size)
                        batch_input = batch_input.reshape((batch_input.shape[0],1,batch_input.shape[1],1))
                        feed_dict = {self.net_input: batch_input, self.label_batch: batch_labels, self.keep_prob: 0.5, self.curr_step : epoch, self.isTrain: True }
                        loss_value, _,  pred,lr= self.sess.run([self.reduced_loss, self.train_optimizer, self.test_prediction,self.learning_rate],feed_dict=feed_dict)
                        if epoch < 5:
                            print('epoch {:d} \t fold = {:d},loss - {:.3f},'.format(epoch, fold_index,loss_value))

                        train_offset = train_offset + self.conf.batch_size

                    #cross validation
                    valid_offset = 0
                    valid_error_rate_list = []
                    while valid_offset < self.reader.get_valid_length():
                        (valid_input, valid_labels) = self.reader.get_validation_batch(fold_index,valid_offset,self.conf.valid_batch_size)
                        valid_input = valid_input.reshape((valid_input.shape[0], 1, valid_input.shape[1], 1))
                        feed_dict = {self.net_input: valid_input, self.label_batch: valid_labels, self.keep_prob: 1.0,self.isTrain: False}
                        valid_pred = self.sess.run([self.test_prediction], feed_dict=feed_dict)
                        valid_pred = np.squeeze(np.asarray(valid_pred))
                        epoc_error_rate_list.append(self.predictions_error(valid_pred, valid_labels))
                        valid_error_rate_list.append(self.predictions_error(valid_pred, valid_labels))
                        valid_offset = valid_offset + self.conf.valid_batch_size
                    if epoch < 5:
                        print('fold validattion error')
                        print(np.array(valid_error_rate_list).mean())

                mean_error = np.array(epoc_error_rate_list).mean()
                duration = time.time() - start_time
                print('step {:d} \t loss = {:.3f},mean_error = {:.3f}, ({:.3f} sec/step)'.format(epoch, loss_value, mean_error, duration))
                write_log('{:d}, {:.3f}, {:.3f}'.format(epoch, loss_value, mean_error), self.conf.logfile)
                if epoch > 0:
                    if epoch % self.conf.save_interval == 0:
                        self.save(self.saver, epoch)
        else:
            # testing on 10 crops of each sample
            test_error_rate_list = []
            for fold_index in range(self.conf.k_cross_val):
                fold_sum = 0

                for mm in range(self.reader.get_valid_length()):
                    (crops_mat, crop_label) = self.reader.get_validation_samples_of_1_input(fold_index, mm)
                    crops_mat = crops_mat.reshape((crops_mat.shape[0], 1, crops_mat.shape[1], 1))
                    feed_dict = {self.net_input: crops_mat, self.label_batch: crop_label, self.keep_prob: 1.0, self.isTrain: False}
                    test_pred = self.sess.run([self.test_prediction], feed_dict=feed_dict)
                    test_pred = np.squeeze(np.asarray(test_pred))
                    # averaging over 10 predictions to get the final predication of this sample
                    sample_pred = np.average(test_pred, axis=0)
                    if np.argmax(sample_pred) == np.argmax(crop_label[0, :]):
                        fold_sum += 1
                fold_error_rate =100-100*(fold_sum/self.reader.get_valid_length())
                print('testing fold {:d} \t error = {:.3f}'.format(fold_index, fold_error_rate))
                write_log('testing fold {:d} \t error = {:.3f}'.format(fold_index, fold_error_rate), self.conf.logfile)
                test_error_rate_list.append(fold_error_rate)
            print('total error = {:.3f}'.format(np.array(test_error_rate_list).mean()))
            write_log('total error = {:.3f}'.format(np.array(test_error_rate_list).mean()), self.conf.logfile)





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
        self.saver = tf.train.Saver(var_list=tf.global_variables())

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
        checkpoint_path = os.path.join(self.conf.modeldir, model_name)
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
    
    
    
    
    
    
    
    
    
    
    
