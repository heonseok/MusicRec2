import tensorflow as tf
import numpy as np
import os
import logging

class AE():
    def __init__(self, logger, gpu_id, learning_rate, input_dim, ae_h_dim_list, z_dim):
        self.logger = logger
        self.gpu_id = gpu_id

        self.learning_rate = learning_rate 

        self.input_dim = input_dim
        self.z_dim = z_dim

        self.enc_h_dim_list = [*ae_h_dim_list, z_dim]
        self.dec_h_dim_list = [*list(reversed(ae_h_dim_list))]

        self.build_model()

    def build_model(self):
        with tf.device('/gpu:%d' % self.gpu_id):
            self.X = tf.placeholder(tf.float32, [None, self.input_dim])

            previous_layer = self.X
            for idx, enc_h_dim in enumerate(self.enc_h_dim_list):
                #print(idx, enc_h_dim)
                previous_layer = tf.layers.dense(inputs=previous_layer, units=enc_h_dim, activation=tf.nn.relu)

            for idx, dec_h_dim in enumerate(self.dec_h_dim_list):
                #print(idx, dec_h_dim)
                previous_layer = tf.layers.dense(inputs=previous_layer, units=dec_h_dim, activation=tf.nn.relu)

            output = tf.layers.dense(inputs=previous_layer, units=self.input_dim, activation=tf.nn.tanh) #, kernel_initializer=tf.contrib.layers.xavier_initializer)
            #cost = tf.reduce_mean(tf.square(X-output))
            self.cost = tf.losses.mean_squared_error(self.X, output)
            #cost_summary = tf.summary.scalar('cost', cost)
            self.solver = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)

    def train(self, sess, batch_xs, epoch_idx, batch_idx, train_batch_total, log_flag):
        _, cost_val = sess.run([self.solver, self.cost], feed_dict={self.X: batch_xs})
        if log_flag == True:
            self.logger.debug('Epoch %.3i, Batch[%.3i/%i], Train loss: %.4E' % (epoch_idx, batch_idx + 1, train_batch_total, cost_val))
        #ogger.debug('Epoch %.3i, Train loss: %.4E' % (epoch_idx+1, train_total_cost / train_batch_total))
        return cost_val
        

    def inference(self, sess, batch_xs, epoch_idx, batch_idx, train_batch_total, log_flag):
        cost_val = sess.run(self.cost, feed_dict={self.X: batch_xs})
        if log_flag == True:
            self.logger.debug('Epoch %.3i, Batch[%.3i/%i], Train loss: %.4E' % (epoch_idx, batch_idx + 1, train_batch_total, cost_val))
        return cost_val

