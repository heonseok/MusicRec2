import tensorflow as tf
import numpy as np
import os
import logging
from utils import sample_z
from utils import kl_divergence_normal_distribution

class VAE_GAN():
    def __init__(self, logger, gpu_id, learning_rate, input_dim, ae_h_dim_list, z_dim, dis_h_dim_list):
        self.logger = logger
        self.gpu_id = gpu_id

        self.learning_rate = learning_rate 

        self.input_dim = input_dim
        self.z_dim = z_dim

        self.enc_h_dim_list = ae_h_dim_list
        self.dec_h_dim_list = [*list(reversed(ae_h_dim_list))]
        self.dis_h_dim_list = dis_h_dim_list

        self.build_model()

    def build_model(self):
        with tf.device('/gpu:%d' % self.gpu_id):
            self.X = tf.placeholder(tf.float32, [None, self.input_dim])
            self.k = tf.placeholder(tf.int32)

            """
            with tf.variable_scope('encoder') as sceop:
                previous_layer = self.X
                for idx, enc_h_dim in enumerate(self.enc_h_dim_list):
                    #print(idx, enc_h_dim)
                    previous_layer = tf.layers.dense(inputs=previous_layer, units=enc_h_dim, activation=tf.nn.relu, name='Enc_h%d'%enc_h_dim)

                self.z_mu = tf.layers.dense(inputs=previous_layer, units=self.z_dim, activation=None, name='Enc_zmu%d'%self.z_dim)
                self.z_logvar = tf.layers.dense(inputs=previous_layer, units=self.z_dim, activation=None, name='Enc_zlogvar%d'%self.z_dim)
     
                self.z = sample_z(self.z_mu, self.z_logvar)
            """
            self.z_mu, self.z_logvar = self.encoder(self.X, self.enc_h_dim_list, self.z_dim)
            self.z = sample_z(self.z_mu, self.z_logvar)

            recon_X = self.decoder(self.z, self.dec_h_dim_list, self.input_dim, False)
            gen_X = self.decoder(tf.random_normal(tf.shape(self.z)), self.dec_h_dim_list, self.input_dim, True)
            """
            with tf.variable_scope('decoder') as scope:
                previous_layer = self.z
                for idx, dec_h_dim in enumerate(self.dec_h_dim_list):
                    #print(idx, dec_h_dim)
                    previous_layer = tf.layers.dense(inputs=previous_layer, units=dec_h_dim, activation=tf.nn.relu, name='Dec_h%d'%dec_h_dim)

                recon_X = tf.layers.dense(inputs=previous_layer, units=self.input_dim, activation=tf.nn.tanh, name='Dec_r%d'%self.input_dim) #, kernel_initializer=tf.contrib.layers.xavier_initializer)
            
                scope.reuse_variables()
            
                previous_layer = tf.random_normal(tf.shape(self.z))
                for idx, dec_h_dim in enumerate(self.dec_h_dim_list):
                    #print(idx, dec_h_dim)
                    previous_layer = tf.layers.dense(inputs=previous_layer, units=dec_h_dim, activation=tf.nn.relu, name='Dec_h%d'%dec_h_dim)

                gen_X = tf.layers.dense(inputs=previous_layer, units=self.input_dim, activation=tf.nn.tanh, name='Dec_r%d'%self.input_dim) #, kernel_initializer=tf.contrib.layers.xavier_initializer)

            """
            with tf.variable_scope('discriminator') as scope:
                previous_layer = recon_X 
                for idx, dis_h_dim in enumerate(self.dis_h_dim_list):
                    #print(idx, dec_h_dim)
                    previous_layer = tf.layers.dense(inputs=previous_layer, units=dis_h_dim, activation=tf.nn.relu, name='Dis_h%d'%dis_h_dim)

                dis_logit_real = tf.layers.dense(inputs=previous_layer, units=1, activation=None, name='Dis_o1') #, kernel_initializer=tf.contrib.layers.xavier_initializer)
                dis_prob_real = tf.nn.sigmoid(dis_logit_real)
            
                scope.reuse_variables()
            
                previous_layer = gen_X 
                for idx, dis_h_dim in enumerate(self.dis_h_dim_list):
                    #print(idx, dec_h_dim)
                    previous_layer = tf.layers.dense(inputs=previous_layer, units=dis_h_dim, activation=tf.nn.relu, name='Dis_h%d'%dis_h_dim)

                dis_logit_fake = tf.layers.dense(inputs=previous_layer, units=1, activation=None, name='Dis_o1') #, kernel_initializer=tf.contrib.layers.xavier_initializer)
                dis_prob_fake = tf.nn.sigmoid(dis_logit_fake)

            self.logger.info([x.name for x in tf.global_variables()])
            print([x.name for x in tf.global_variables() if 'enc' in x.name])
            print([x.name for x in tf.global_variables() if 'dec' in x.name])
            print([x.name for x in tf.global_variables() if 'dis' in x.name])

            enc_theta = ([x for x in tf.global_variables() if 'enc' in x.name])
            dec_theta = ([x for x in tf.global_variables() if 'dec' in x.name])
            dis_theta = ([x for x in tf.global_variables() if 'dis' in x.name])

            #cost = tf.reduce_mean(tf.square(X-output))
            self.recon_loss = tf.losses.mean_squared_error(self.X, recon_X)
            self.kl_loss = kl_divergence_normal_distribution(self.z_mu, self.z_logvar)
            
            self.dec_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logit_fake, labels=tf.ones_like(dis_logit_fake)))

            self.dis_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logit_real, labels=tf.ones_like(dis_logit_real)))
            self.dis_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logit_fake, labels=tf.zeros_like(dis_logit_fake)))

            self.enc_loss = self.recon_loss + self.kl_loss
            self.dec_loss = self.dec_loss_fake + self.recon_loss 
            self.dis_loss = self.dis_loss_real + self.dis_loss_fake
            #cost_summary = tf.summary.scalar('cost', cost)

            self.total_loss = self.enc_loss + self.dec_loss + self.dis_loss 
            #self.solver = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.total_loss)
            self.enc_solver = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.enc_loss, var_list=enc_theta)
            self.dec_solver = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.dec_loss, var_list=dec_theta)
            self.dis_solver = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.dis_loss, var_list=dis_theta)

            ### Recommendaiton metric ###
        with tf.device('/cpu:0'):
            self.top_k_op = tf.nn.top_k(recon_X, self.k)

    def encoder(self, X, enc_h_dim_list, z_dim):
        with tf.variable_scope('encoder') as sceop:
            previous_layer = X
            for idx, enc_h_dim in enumerate(enc_h_dim_list):
                #print(idx, enc_h_dim)
                previous_layer = tf.layers.dense(inputs=previous_layer, units=enc_h_dim, activation=tf.nn.relu, name='h%d'%enc_h_dim)

            z_mu = tf.layers.dense(inputs=previous_layer, units=z_dim, activation=None, name='zmu%d'%z_dim)
            z_logvar = tf.layers.dense(inputs=previous_layer, units=z_dim, activation=None, name='zlogvar%d'%z_dim)
 
            return z_mu, z_logvar 
          
    def decoder(self, z, dec_h_dim_list, dec_dim, reuse_flag):
        with tf.variable_scope('decoder') as scope:
            if reuse_flag == True:
                scope.reuse_variables()

            previous_layer = z
            for idx, dec_h_dim in enumerate(dec_h_dim_list):
                #print(idx, dec_h_dim)
                previous_layer = tf.layers.dense(inputs=previous_layer, units=dec_h_dim, activation=tf.nn.relu, name='h%d'%dec_h_dim)

            dec_X = tf.layers.dense(inputs=previous_layer, units=dec_dim, activation=tf.nn.tanh, name='dec%d'%self.input_dim) 

            return dec_X
        
    def train(self, sess, batch_xs, epoch_idx, batch_idx, batch_total, log_flag):
        _, dis_loss_val = sess.run([self.dis_solver, self.dis_loss], feed_dict={self.X: batch_xs})
        _, dec_loss_val = sess.run([self.dec_solver, self.dec_loss], feed_dict={self.X: batch_xs})
        _, enc_loss_val = sess.run([self.enc_solver, self.enc_loss], feed_dict={self.X: batch_xs})
        total_loss_val = dis_loss_val + dec_loss_val + enc_loss_val

        if log_flag == True:
            self.logger.debug('Epoch %.3i, Batch[%.3i/%i], Enc loss : %.4E, Dec loss : %.4E, Dec loss : %.4E, Train loss: %.4E' % (epoch_idx, batch_idx + 1, batch_total, enc_loss_val, dec_loss_val, dis_loss_val, total_loss_val))
        return total_loss_val
        

    def inference(self, sess, batch_xs, epoch_idx, batch_idx, batch_total, log_flag):
        enc_loss_val, dec_loss_val, dis_loss_val = sess.run([self.enc_loss, self.dec_loss, self.dis_loss], feed_dict={self.X: batch_xs})
        total_loss_val = dis_loss_val + dec_loss_val + enc_loss_val

        if log_flag == True:
            self.logger.debug('Epoch %.3i, Batch[%.3i/%i], Enc loss : %.4E, Dec loss : %.4E, Dec loss : %.4E, Valid loss: %.4E' % (epoch_idx, batch_idx + 1, batch_total, enc_loss_val, dec_loss_val, dis_loss_val, total_loss_val))
        return total_loss_val

    def inference_with_top_k(self, sess, batch_xs, epoch_idx, batch_idx, batch_total, log_flag, k):
        enc_loss_val, dec_loss_val, dis_loss_val, top_k = sess.run([self.enc_loss, self.dec_loss, self.dis_loss, self.top_k_op], feed_dict={self.X: batch_xs, self.k: k})
        total_loss_val = dis_loss_val + dec_loss_val + enc_loss_val

        if log_flag == True:
            self.logger.debug('Epoch %.3i, Batch[%.3i/%i], Enc loss : %.4E, Dec loss : %.4E, Dec loss : %.4E, Test loss: %.4E' % (epoch_idx, batch_idx + 1, batch_total, enc_loss_val, dec_loss_val, dis_loss_val, total_loss_val))
        return total_loss_val, top_k
