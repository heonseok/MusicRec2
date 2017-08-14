import tensorflow as tf
import numpy as np
import os
import logging
from utils import sample_z
from utils import kl_divergence_normal_distribution

class VAE():
    def __init__(self, logger, gpu_id, learning_rate, input_dim, ae_h_dim_list, z_dim):
        self.logger = logger
        self.gpu_id = gpu_id

        self.learning_rate = learning_rate 

        self.input_dim = input_dim
        self.z_dim = z_dim

        self.enc_h_dim_list = [*ae_h_dim_list]
        #self.enc_h_dim_list = [*ae_h_dim_list, z_dim]
        self.dec_h_dim_list = [*list(reversed(ae_h_dim_list))]

        self.build_model()

    def build_model(self):
        with tf.device('/gpu:%d' % self.gpu_id):
            self.X = tf.placeholder(tf.float32, [None, self.input_dim])
            self.k = tf.placeholder(tf.int32)

            previous_layer = self.X
            for idx, enc_h_dim in enumerate(self.enc_h_dim_list):
                #print(idx, enc_h_dim)
                previous_layer = tf.layers.dense(inputs=previous_layer, units=enc_h_dim, activation=tf.nn.relu)

            self.z_mu = tf.layers.dense(inputs=previous_layer, units=self.z_dim, activation=None)
            self.z_logvar = tf.layers.dense(inputs=previous_layer, units=self.z_dim, activation=None)
 
            self.z = sample_z(self.z_mu, self.z_logvar)

            previous_layer = self.z
            for idx, dec_h_dim in enumerate(self.dec_h_dim_list):
                #print(idx, dec_h_dim)
                previous_layer = tf.layers.dense(inputs=previous_layer, units=dec_h_dim, activation=tf.nn.relu)

            output = tf.layers.dense(inputs=previous_layer, units=self.input_dim, activation=tf.nn.tanh) #, kernel_initializer=tf.contrib.layers.xavier_initializer)
            #cost = tf.reduce_mean(tf.square(X-output))
            self.recon_loss = tf.losses.mean_squared_error(self.X, output)
            self.kl_loss = kl_divergence_normal_distribution(self.z_mu, self.z_logvar)
            #cost_summary = tf.summary.scalar('cost', cost)
            self.total_loss = self.recon_loss + self.kl_loss
            self.solver = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.total_loss)

            ### Recommendaiton metric ###
        with tf.device('/cpu:0'):
            self.top_k_op = tf.nn.top_k(output, self.k)

    def train(self, sess, batch_xs, epoch_idx, batch_idx, train_batch_total, log_flag):
        _, loss_val = sess.run([self.solver, self.total_loss], feed_dict={self.X: batch_xs})
        if log_flag == True:
            self.logger.debug('Epoch %.3i, Batch[%.3i/%i], Train loss: %.4E' % (epoch_idx, batch_idx + 1, train_batch_total, loss_val))
        #ogger.debug('Epoch %.3i, Train loss: %.4E' % (epoch_idx+1, train_total_loss / train_batch_total))
        return loss_val
        

    def inference(self, sess, batch_xs, epoch_idx, batch_idx, batch_total, log_flag):
        loss_val = sess.run(self.total_loss, feed_dict={self.X: batch_xs})
        if log_flag == True:
            self.logger.debug('Epoch %.3i, Batch[%.3i/%i], Train loss: %.4E' % (epoch_idx, batch_idx + 1, batch_total, loss_val))
        return loss_val

    def inference_with_top_k(self, sess, batch_xs, epoch_idx, batch_idx, batch_total, log_flag, k):
        loss_val, top_k = sess.run([self.total_loss, self.top_k_op], feed_dict={self.X: batch_xs, self.k: k})
        if log_flag == True:
            self.logger.debug('Epoch %.3i, Batch[%.3i/%i], Train loss: %.4E' % (epoch_idx, batch_idx + 1, batch_total, loss_val))
        return loss_val, top_k
