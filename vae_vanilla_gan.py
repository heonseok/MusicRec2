import tensorflow as tf
import numpy as np
from base_model import BaseModel
from utils import sample_z
from utils import kl_divergence_normal_distribution

class VAE_VANILLA_GAN(BaseModel):
    def __init__(self, logger, gpu_id, learning_rate, loss_type, input_dim, z_dim, ae_h_dim_list, dis_h_dim_list):
        super(VAE_VANILLA_GAN, self).__init__(logger, gpu_id, learning_rate, loss_type, input_dim, z_dim) 

        self.enc_h_dim_list = ae_h_dim_list
        self.dec_h_dim_list = [*list(reversed(ae_h_dim_list))]
        self.dis_h_dim_list = dis_h_dim_list

        self.build_model()

    def build_model(self):
        with tf.device('/gpu:%d' % self.gpu_id):
            ### Placeholder ###
            self.X = tf.placeholder(tf.float32, [None, self.input_dim])
            self.k = tf.placeholder(tf.int32)
            self.z = tf.placeholder(tf.float32, [None, self.z_dim])
            self.keep_prob = tf.placeholder(tf.float32)

            ### Encoding ###
            self.z_mu, self.z_logvar = self.encoder(self.X, self.enc_h_dim_list, self.z_dim, self.keep_prob)
            self.z_sampled = sample_z(self.z_mu, self.z_logvar)

            self.kl_loss = kl_divergence_normal_distribution(self.z_mu, self.z_logvar)

            ### Decoding ###
            self.recon_X_logit = self.decoder(self.z_sampled, self.dec_h_dim_list, self.input_dim, self.keep_prob, False)
            self.recon_X = tf.nn.sigmoid(self.recon_X_logit)
            self.recon_X_display = tf.nn.tanh(self.recon_X_logit)

            self.recon_loss = self.recon_loss() 

            ### Generating ###
            self.gen_X_logit = self.decoder(self.z, self.dec_h_dim_list, self.input_dim, self.keep_prob, True)
            self.gen_X = tf.nn.sigmoid(self.gen_X_logit)

            ### Discriminating ###
            dis_logit_real, _ = self.discriminator(self.X, self.dis_h_dim_list, 1, self.keep_prob, False)
            dis_logit_fake, _ = self.discriminator(self.gen_X, self.dis_h_dim_list, 1, self.keep_prob, True)

            self.dec_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logit_fake, labels=tf.ones_like(dis_logit_fake))) 

            self.dis_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logit_real, labels=tf.ones_like(dis_logit_real))) 
            self.dis_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logit_fake, labels=tf.zeros_like(dis_logit_fake))) 
            

            ### Loss ###
            self.enc_loss = self.kl_loss + self.recon_loss 
            self.dec_loss = self.recon_loss + self.dec_loss_fake 
            self.dis_loss = self.dis_loss_real + self.dis_loss_fake
            
            ### Theta ###
            enc_theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='enc')
            dec_theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dec')
            dis_theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dis')
            #dec_theta = ([x for x in tf.global_variables() if 'dec' in x.name])
            #dis_theta = ([x for x in tf.global_variables() if 'dis' in x.name])

            ### Solver ###
            self.enc_solver = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.enc_loss, var_list=enc_theta)
            self.dec_solver = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.dec_loss, var_list=dec_theta)
            self.dis_solver = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.dis_loss, var_list=dis_theta)

            #self.enc_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.enc_loss, var_list=enc_theta)
            #self.dec_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.dec_loss, var_list=dec_theta)
            #self.dis_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.dis_loss, var_list=dis_theta)

    def train(self, sess, batch_xs, epoch_idx, batch_idx, batch_total, log_flag, keep_prob):
         _, dis_loss_val = sess.run([self.dis_solver, self.dis_loss], feed_dict={self.X: batch_xs, self.keep_prob: keep_prob, self.z: np.random.normal(0,1, size=[256, self.z_dim])})
         _, dec_loss_val = sess.run([self.dec_solver, self.dec_loss], feed_dict={self.X: batch_xs, self.keep_prob: keep_prob, self.z: np.random.normal(0,1, size=[256, self.z_dim])})
         _, enc_loss_val = sess.run([self.enc_solver, self.enc_loss], feed_dict={self.X: batch_xs, self.keep_prob: keep_prob})

         total_loss_val = dis_loss_val + dec_loss_val + enc_loss_val
         return total_loss_val

    def inference(self, sess, batch_xs, epoch_idx, batch_idx, batch_total, log_flag, keep_prob):
         dis_loss_val = sess.run(self.dis_loss, feed_dict={self.X: batch_xs, self.keep_prob: keep_prob, self.z: np.random.normal(0,1, size=[256, self.z_dim])})
         dec_loss_val = sess.run(self.dec_loss, feed_dict={self.X: batch_xs, self.keep_prob: keep_prob, self.z: np.random.normal(0,1, size=[256, self.z_dim])})
         enc_loss_val = sess.run(self.enc_loss, feed_dict={self.X: batch_xs, self.keep_prob: keep_prob})

         total_loss_val = dis_loss_val + dec_loss_val + enc_loss_val
         return total_loss_val

    def inference_with_recon(self, sess, batch_xs, epoch_idx, batch_idx, batch_total, log_flag, keep_prob):
         dis_loss_val = sess.run(self.dis_loss, feed_dict={self.X: batch_xs, self.keep_prob: keep_prob, self.z: np.random.normal(0,1, size=[256, self.z_dim])})
         dec_loss_val = sess.run(self.dec_loss, feed_dict={self.X: batch_xs, self.keep_prob: keep_prob, self.z: np.random.normal(0,1, size=[256, self.z_dim])})
         enc_loss_val = sess.run(self.enc_loss, feed_dict={self.X: batch_xs, self.keep_prob: keep_prob})
         recon_val = sess.run(self.recon_X_display, feed_dict={self.X: batch_xs, self.keep_prob: keep_prob})
         total_loss_val = dis_loss_val + dec_loss_val + enc_loss_val

         return total_loss_val, recon_val
