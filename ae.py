import tensorflow as tf
import numpy as np

import logging

flags = tf.app.flags

flags.DEFINE_string("datset", "MNIST", "Dataset (MNIST,Music) for experiment [MNIST]")

flags.DEFINE_integer("epoch", 50, "Epoch to train [50]")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate [0.01]")
flags.DEFINE_integer("batch_size", 100, "Batch size [100]")
#flags.DEFINE_integer("batch_logging_step", 10, "Logging step for batch [100]")
#flags.DEFINE_integer("epoch_logging_step", 1, "Logging step for epoch [1]")  # Need?

flags.DEFINE_integer("input_dim", 28*28, "Dimension of input [28*28]")
flags.DEFINE_string("ae_h_dim_list", "[256]", "List of AE dimensions [256]")
flags.DEFINE_integer("z_dim", 128, "Dimension of z [128]")

flags.DEFINE_integer("gpu_id", 2, "GPU id [0]")
flags.DEFINE_string("data_dir", "data", 'Directory name to load input data [data]')
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("log_dir", "log", "Directory name to save the logs [log]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [Fasle]")
flags.DEFINE_boolean("continue_train", None,
                     "True to continue training from saved checkpoint. False for restarting. None for automatic [None]")
FLAGS = flags.FLAGS

if FLAGS.datset == "MNIST":
    import matplotlib.pyplot as plt
    from utils import Drawer
    drawer = Drawer()
    draw_flag = True

    from tensorflow.examples.tutorials.mnist import input_data
    data = input_data.read_data_sets("MNIST/data/", one_hot=True)

elif FLAGS.dataset == "Music":
    import data_loader
    draw_flag = False
    data = data_loader.load_music_data()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

sh = logging.StreamHandler()
fh = logging.FileHandler(FLAGS.log_dir)

fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
sh.setFormatter(fmt)
fh.setFormatter(fmt)

logger.addHandler(sh)
logger.addHandler(fh)



def main(_):
    logger.info("Running AE")

    with tf.device('/gpu:%d' % FLAGS.gpu_id):
        X = tf.placeholder(tf.float32, [None, FLAGS.input_dim])
        ae_h_dim_list = eval(FLAGS.ae_h_dim_list)
        enc_h_dim_list = [*ae_h_dim_list, FLAGS.z_dim]
        dec_h_dim_list = [*list(reversed(ae_h_dim_list))]

        previous_layer = X
        for idx, enc_h_dim in enumerate(enc_h_dim_list):
            print(idx, enc_h_dim)
            previous_layer = tf.layers.dense(inputs=previous_layer, units=enc_h_dim, activation=tf.nn.relu)

        for idx, dec_h_dim in enumerate(dec_h_dim_list):
            print(idx, dec_h_dim)
            previous_layer = tf.layers.dense(inputs=previous_layer, units=dec_h_dim, activation=tf.nn.relu)

        output = tf.layers.dense(inputs=previous_layer, units=FLAGS.input_dim) #, kernel_initializer=tf.contrib.layers.xavier_initializer)
        cost = tf.losses.mean_squared_error(X, output)
        solver = tf.train.RMSPropOptimizer(FLAGS.learning_rate).minimize(cost)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True


        with tf.Session(config=config) as sess:
            sess.run(init)

            train_batch_total = int(data.train.num_examples / FLAGS.batch_size)
            valid_batch_total = int(data.validation.num_examples/ FLAGS.batch_size)

            train_logging_step = train_batch_total/10;
            valid_logging_step = valid_batch_total/10;

            for epoch_idx in range(FLAGS.epoch):
                train_total_cost = 0
                valid_total_cost = 0

                for batch_idx in range(train_batch_total):
                    batch_xs, batch_ys = data.train.next_batch(FLAGS.batch_size)
                    _, cost_val = sess.run([solver, cost],
                                           feed_dict={X: batch_xs})
                    train_total_cost += cost_val
                    if((batch_idx+1) % train_logging_step == 0):
                    #if (batch_idx % FLAGS.batch_logging_step == 0):
                        logger.debug('Epoch %.3i, Batch[%.3i/%i], Train loss: %.4E' % (epoch_idx + 1, batch_idx + 1, train_batch_total, cost_val))
                logger.debug('Epoch %.3i, Train loss: %.4E' % (epoch_idx+1, train_total_cost/train_batch_total))


                for batch_idx in range(valid_batch_total):
                    batch_xs, batch_ys = data.validation.next_batch(FLAGS.batch_size)
                    _, cost_val = sess.run([solver, cost],
                                           feed_dict={X: batch_xs})
                    valid_total_cost += cost_val
                    if ((batch_idx+1) % valid_logging_step == 0):
                    #if (batch_idx % FLAGS.batch_logging_step == 0):
                        logger.debug('Epoch %.3i, Batch[%.3i/%i], Valid loss: %.4E' % (epoch_idx + 1, batch_idx + 1, valid_batch_total, cost_val))
                logger.debug('Epoch %.3i, Valid loss: %.4E' % (epoch_idx + 1, valid_total_cost / valid_batch_total))


                if draw_flag == True:
                    sample_size = 16
                    samples = sess.run(output, feed_dict={X: data.test.images[:sample_size]})
                    fig = drawer.plot(samples)
                    plt.savefig('out_ae/{}.png'.format(str(epoch_idx+1).zfill(3)), bbox_inches='tight')


if __name__ == '__main__':
    tf.app.run()


