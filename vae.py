import tensorflow as tf
import numpy as np
import os
import logging
from utils import sample_z
from utils import kl_divergence_normal_distribution
from base_model import BaseModel

class VAE(BaseModel):
    def __init__(self, logger, gpu_id, learning_rate, input_dim, z_dim, ae_h_dim_list):
        super(VAE, self).__init__(logger, gpu_id, learning_rate, input_dim, z_dim)

        self.enc_h_dim_list = ae_h_dim_list
        self.dec_h_dim_list = [*list(reversed(ae_h_dim_list))]

        self.build_model()

    def build_model(self):
        with tf.device('/gpu:%d' % self.gpu_id):
            self.X = tf.placeholder(tf.float32, [None, self.input_dim])
            self.k = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32)

            ### Encoding ###
            self.z_mu, self.z_logvar = self.encoder(self.X, self.enc_h_dim_list, self.z_dim, self.keep_prob)
            self.z = sample_z(self.z_mu, self.z_logvar)
 
            ### Decoding ### 
            self.recon_X_logit = self.decoder(self.z, self.dec_h_dim_list, self.input_dim, self.keep_prob, False)

            self.logger.info([x.name for x in tf.global_variables()])
            #cost = tf.reduce_mean(tf.square(X-output))

            ### Loss ###
            self.recon_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.recon_X_logit, labels=self.X))
            #self.recon_loss = tf.losses.mean_squared_error(self.X, self.recon_X_prob)
            self.kl_loss = kl_divergence_normal_distribution(self.z_mu, self.z_logvar)
            #cost_summary = tf.summary.scalar('cost', cost)
            self.total_loss = self.recon_loss + self.kl_loss
            self.solver = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.total_loss)

            ### Recommendaiton metric ###
            self.recon_X = tf.nn.tanh(self.recon_X_logit)
            #self.recon_X = tf.nn.sigmoid(self.recon_X_logit)
        with tf.device('/cpu:0'):
            self.top_k_op = tf.nn.top_k(self.recon_X, self.k)

    def train(self, sess, batch_xs, epoch_idx, batch_idx, train_batch_total, log_flag, keep_prob):
        _, total_loss_val, recon_loss_val, kl_loss_val = sess.run([self.solver, self.total_loss, self.recon_loss, self.kl_loss], feed_dict={self.X: batch_xs, self.keep_prob: keep_prob})
        if log_flag == True:
            self.logger.debug('Epoch %.3i, Batch[%.3i/%i], Recon loss : %.4E, KL loss : %.4E, Train loss: %.4E' % (epoch_idx, batch_idx + 1, train_batch_total, recon_loss_val, kl_loss_val, total_loss_val))
        return total_loss_val
        

    def inference(self, sess, batch_xs, epoch_idx, batch_idx, batch_total, log_flag, keep_prob):
        total_loss_val, recon_loss_val, kl_loss_val = sess.run([self.total_loss, self.recon_loss, self.kl_loss], feed_dict={self.X: batch_xs, self.keep_prob: keep_prob})

        if log_flag == True:
            self.logger.debug('Epoch %.3i, Batch[%.3i/%i], Recon loss : %.4E, KL loss : %.4E, Valid loss: %.4E' % (epoch_idx, batch_idx + 1, batch_total, recon_loss_val, kl_loss_val, total_loss_val))
        return total_loss_val

    def inference_with_top_k(self, sess, batch_xs, epoch_idx, batch_idx, batch_total, log_flag, keep_prob, k):
        total_loss_val, recon_loss_val, kl_loss_val, top_k = sess.run([self.total_loss, self.recon_loss, self.kl_loss, self.top_k_op], feed_dict={self.X: batch_xs, self.keep_prob: keep_prob, self.k: k})
        if log_flag == True:
            self.logger.debug('Epoch %.3i, Batch[%.3i/%i], Recon loss : %.4E, KL loss : %.4E, Test loss: %.4E' % (epoch_idx, batch_idx + 1, batch_total, recon_loss_val, kl_loss_val, total_loss_val))
        return total_loss_val, top_k

    def inference_with_recon(self, sess, batch_xs, epoch_idx, batch_idx, batch_total, log_flag, keep_prob):
        total_loss_val, recon_loss_val, kl_loss_val, recon_val = sess.run([self.total_loss, self.recon_loss, self.kl_loss, self.recon_X], feed_dict={self.X: batch_xs, self.keep_prob: keep_prob})
        if log_flag == True:
            self.logger.debug('Epoch %.3i, Batch[%.3i/%i], Recon loss : %.4E, KL loss : %.4E, Test loss: %.4E' % (epoch_idx, batch_idx + 1, batch_total, recon_loss_val, kl_loss_val, total_loss_val))
        return total_loss_val, recon_val 
