import tensorflow as tf 

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)

class BaseModel():
    def __init__(self, gpu_id, learning_rate, loss_type, input_dim, z_dim):
        self.gpu_id = gpu_id

        self.learning_rate = learning_rate 

        self.input_dim = input_dim
        self.z_dim = z_dim

        self.w_init = tf.contrib.layers.variance_scaling_initializer()

        self.loss_type = loss_type

    def encoder(self, X, enc_h_dim_list, z_dim, keep_prob):
        with tf.variable_scope('enc') as sceop:
            previous_layer = X
            for idx, enc_h_dim in enumerate(enc_h_dim_list):
                #print(idx, enc_h_dim)
                #previous_layer = tf.layers.dense(inputs=previous_layer, units=enc_h_dim, activation=tf.nn.relu, kernel_initializer=self.w_init, name='h%d'%enc_h_dim)
                previous_layer = tf.layers.dense(inputs=previous_layer, units=enc_h_dim, activation=None, kernel_initializer=self.w_init, name='h%d'%enc_h_dim)
                previous_layer = lrelu(previous_layer)
                previous_layer = tf.nn.dropout(previous_layer, keep_prob)


            z_mu = tf.layers.dense(inputs=previous_layer, units=z_dim, activation=None, name='zmu%d'%z_dim)
            z_logvar = tf.layers.dense(inputs=previous_layer, units=z_dim, activation=tf.nn.softplus, name='zlogvar%d'%z_dim)
            #z_logvar = tf.layers.dense(inputs=previous_layer, units=z_dim, activation=None, name='zlogvar%d'%z_dim)
 
            return z_mu, z_logvar 
          
    def decoder(self, z, dec_h_dim_list, dec_dim, keep_prob, reuse_flag):
        with tf.variable_scope('dec') as scope:
            if reuse_flag == True:
                scope.reuse_variables()

            previous_layer = z
            for idx, dec_h_dim in enumerate(dec_h_dim_list):
                #print(idx, dec_h_dim)
                #previous_layer = tf.layers.dense(inputs=previous_layer, units=dec_h_dim, activation=tf.nn.relu, kernel_initializer=self.w_init, name='h%d'%dec_h_dim)
                previous_layer = tf.layers.dense(inputs=previous_layer, units=dec_h_dim, activation=None, kernel_initializer=self.w_init, name='h%d'%dec_h_dim)
                previous_layer = lrelu(previous_layer)
                previous_layer = tf.nn.dropout(previous_layer, keep_prob)

            dec_X = tf.layers.dense(inputs=previous_layer, units=dec_dim, activation=None, name='dec%d'%self.input_dim) 

            return dec_X

    def discriminator(self, X, dis_h_dim_list, dis_dim, keep_prob, reuse_flag):
        with tf.variable_scope('dis') as scope:
            if reuse_flag == True:
                scope.reuse_variables()

            previous_layer = X 
            for idx, dis_h_dim in enumerate(dis_h_dim_list):
                #print(idx, dec_h_dim)
                previous_layer = tf.layers.dense(inputs=previous_layer, units=dis_h_dim, activation=None, kernel_initializer=self.w_init, name='h%d'%dis_h_dim)
                previous_layer = lrelu(previous_layer)
                previous_layer = tf.nn.dropout(previous_layer, keep_prob)

            dis_logit = tf.layers.dense(inputs=previous_layer, units=dis_dim, activation=None, name='dis%d'%dis_dim) 
            dis_prob = tf.nn.sigmoid(dis_logit)

            return dis_logit, dis_prob

    def discriminator_with_intermediate_layer(self, X, dis_h_dim_list, dis_dim, keep_prob, reuse_flag):
        with tf.variable_scope('dis') as scope:
            if reuse_flag == True:
                scope.reuse_variables()

            previous_layer = X 
            for idx, dis_h_dim in enumerate(dis_h_dim_list):
                #print(idx, dec_h_dim)
                previous_layer = tf.layers.dense(inputs=previous_layer, units=dis_h_dim, activation=None, kernel_initializer=self.w_init, name='h%d'%dis_h_dim)
                if idx == len(dis_h_dim_list)-1:
                    intermediate_layer = previous_layer
                previous_layer = lrelu(previous_layer)
                previous_layer = tf.nn.dropout(previous_layer, keep_prob)

            dis_logit = tf.layers.dense(inputs=previous_layer, units=dis_dim, activation=None, name='dis%d'%dis_dim) 
            dis_prob = tf.nn.sigmoid(dis_logit)

            return dis_logit, dis_prob, intermediate_layer

    def recon_loss(self):
        if self.loss_type == 'CE':
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.recon_X_logit, labels=self.X))
        elif self.loss_type == 'MSE':
            return tf.losses.mean_squared_error(self.recon_X, self.X)
