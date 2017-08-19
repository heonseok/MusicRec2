import tensorflow as tf
import numpy as np
import os
import logging

class AE():
    def __init__(self, gpu_id, learning_rate, input_dim, ae_h_dim_list, z_dim):
        self.gpu_id = gpu_id

        self.learning_rate = learning_rate 

        self.input_dim = input_dim
        self.z_dim = z_dim

        self.enc_h_dim_list = ae_h_dim_list
        self.dec_h_dim_list = [*list(reversed(ae_h_dim_list))]

        self.keep_prob = 0.9

        self.build_model()

    def build_model(self):
        with tf.device('/gpu:%d' % self.gpu_id):
            ### Placeholder ###
            self.X = tf.placeholder(tf.float32, [None, self.input_dim])
            self.k = tf.placeholder(tf.int32)

            ### Encoding ###
            previous_layer = self.X
            for idx, enc_h_dim in enumerate(self.enc_h_dim_list):
                #print(idx, enc_h_dim)
                previous_layer = tf.layers.dense(inputs=previous_layer, units=enc_h_dim, activation=tf.nn.relu, name='Enc_h%d'%enc_h_dim)
                previous_layer = tf.nn.dropout(previous_layer, self.keep_prob)

            self.z = tf.layers.dense(inputs=previous_layer, units=self.z_dim, activation=None, name='Enc_z%d'%self.z_dim)

            ### Decoding ###
            previous_layer = self.z
            for idx, dec_h_dim in enumerate(self.dec_h_dim_list):
                #print(idx, dec_h_dim)
                previous_layer = tf.layers.dense(inputs=previous_layer, units=dec_h_dim, activation=tf.nn.relu, name='Dec_h%d'%dec_h_dim)
                previous_layer = tf.nn.dropout(previous_layer, self.keep_prob)

            self.recon_X_logit = tf.layers.dense(inputs=previous_layer, units=self.input_dim, activation=None, name='Dec_r%d'%self.input_dim) 
            self.recon_X = tf.nn.tanh(self.recon_X_logit)
            self.recon_X_prob = tf.nn.sigmoid(self.recon_X_logit)
            #self.logger.info([x.name for x in tf.global_variables()])

            ### Loss ###
            self.total_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.recon_X_logit, labels=self.X))
            #cost_summary = tf.summary.scalar('cost', cost)
            #self.solver = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.total_loss)

            ### Solver ###
            self.solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)

            ### Recommendaiton metric ###
        with tf.device('/cpu:0'):
            self.top_k_op = tf.nn.top_k(self.recon_X, self.k)

    def train(self, logger, sess, batch_xs, epoch_idx, batch_idx, train_batch_total, log_flag):
        _, loss_val = sess.run([self.solver, self.total_loss], feed_dict={self.X: batch_xs})
        if log_flag == True:
            logger.debug('Epoch %.3i, Batch[%.3i/%i], Train loss: %.4E' % (epoch_idx, batch_idx + 1, train_batch_total, loss_val))
        return loss_val
        

    def inference(self, logger, sess, batch_xs, epoch_idx, batch_idx, batch_total, log_flag):
        loss_val = sess.run(self.total_loss, feed_dict={self.X: batch_xs})
        if log_flag == True:
            logger.debug('Epoch %.3i, Batch[%.3i/%i], Valid loss: %.4E' % (epoch_idx, batch_idx + 1, batch_total, loss_val))
        return loss_val

    def inference_with_top_k(self, logger, sess, batch_xs, epoch_idx, batch_idx, batch_total, log_flag, k):
        loss_val, top_k = sess.run([self.total_loss, self.top_k_op], feed_dict={self.X: batch_xs, self.k: k})
        if log_flag == True:
            logger.debug('Epoch %.3i, Batch[%.3i/%i], Test loss: %.4E' % (epoch_idx, batch_idx + 1, batch_total, loss_val))
        return loss_val, top_k

    def inference_with_output(self, logger, sess, batch_xs, epoch_idx, batch_idx, batch_total, log_flag):
        loss_val, recon_val = sess.run([self.total_loss, self.recon_X], feed_dict={self.X: batch_xs})
        if log_flag == True:
            logger.debug('Epoch %.3i, Batch[%.3i/%i], Test loss: %.4E' % (epoch_idx, batch_idx + 1, batch_total, loss_val))
        return loss_val, recon_val 
