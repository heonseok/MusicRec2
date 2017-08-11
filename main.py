import tensorflow as tf
import numpy as np
import os
import logging
from ae import AE


def main(_):
    logger.info("Running AE")
    model = AE(logger, FLAGS.gpu_id, FLAGS.learning_rate, input_dim, eval(FLAGS.ae_h_dim_list), FLAGS.z_dim)

    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.log_device_placement = True

    #merged = tf.summary.merge_all()

    with tf.device('cpu:0'):
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.max_to_keep)

    if FLAGS.is_train == True:
        with tf.Session(config=config) as sess:
            sess.run(init)

            #writer = tf.train.SummaryWriter(tensorboard_dir, sess.graph)
            train_batch_total = int(data.train.num_examples / FLAGS.batch_size)
            valid_batch_total = int(data.validation.num_examples/ FLAGS.batch_size)

            train_logging_step = int(train_batch_total/10);
            valid_logging_step = int(valid_batch_total/10);

            best_valid_total_cost = float('inf')
            best_model_idx = 0
            #best_save_path = ''
            valid_non_improve_count = 0

            for epoch_idx in range(FLAGS.epoch):
                train_total_cost = 0
                valid_total_cost = 0
                
                ##### TRAIN #####
                for batch_idx in range(train_batch_total):
                    if mnist_flag == True:
                        batch_xs, batch_ys = data.train.next_batch(FLAGS.batch_size)
                    elif mnist_flag == False:
                        batch_xs = data.train.next_batch(FLAGS.batch_size)

                    if ((batch_idx+1) % FLAGS.batch_logging_step == 0):
                        #logger.debug('Epoch %.3i, Batch[%.3i/%i], Train loss: %.4E' % (epoch_idx + 1, batch_idx + 1, train_batch_total, cost_val))
                        log_flag = True
                    else:
                        log_flag = False

                    #_, cost_val, summary = sess.run([solver, cost, merged], feed_dict={X: batch_xs})
                    #writer.add_summary(summary, global_step=epoch_idx*tranin_total_batch+batch_idx)
                    cost_val = model.train(sess, batch_xs, epoch_idx, batch_idx, train_batch_total, log_flag)
                    #_, cost_val = sess.run([solver, cost], feed_dict={X: batch_xs})
                    train_total_cost += cost_val

                    #if((batch_idx+1) % train_logging_step == 0):
                logger.debug('Epoch %.3i, Train loss: %.4E' % (epoch_idx+1, train_total_cost / train_batch_total))
                save_path = saver.save(sess, ckpt_path, global_step=epoch_idx) 


                ##### VALIDATION #####
                for batch_idx in range(valid_batch_total):
                    if mnist_flag == True:
                        batch_xs, batch_ys = data.validation.next_batch(FLAGS.batch_size)
                    elif mnist_flag == False:
                        batch_xs = data.validation.next_batch(FLAGS.batch_size)

                    if ((batch_idx+1) % FLAGS.batch_logging_step == 0):
                        #logger.debug('Epoch %.3i, Batch[%.3i/%i], Train loss: %.4E' % (epoch_idx + 1, batch_idx + 1, train_batch_total, cost_val))
                        log_flag = True
                    else:
                        log_flag = False

                    cost_val = model.inference(sess, batch_xs, epoch_idx, batch_idx, valid_batch_total, log_flag)
                    #cost_val = sess.run(cost, feed_dict={X: batch_xs})
                    valid_total_cost += cost_val

                    #if ((batch_idx+1) % valid_logging_step == 0):
                    #if ((batch_idx+1) % FLAGS.batch_logging_step == 0):
                        #logger.debug('Epoch %.3i, Batch[%.3i/%i], Valid loss: %.4E' % (epoch_idx + 1, batch_idx + 1, valid_batch_total, cost_val))
                logger.debug('Epoch %.3i, Valid loss: %.4E' % (epoch_idx + 1, valid_total_cost / valid_batch_total))

                ## Update best_valid_total_cost
                if valid_total_cost < best_valid_total_cost:
                    best_valid_total_cost = valid_total_cost
                    valid_non_improve_count = 0
                    best_model_idx = epoch_idx 
                    #best_save_path = save_path
                else:
                    valid_non_improve_count += 1
                    logger.info("Valid cost has not been improved for %d epochs" % valid_non_improve_count)
                    if valid_non_improve_count == 10:
                        break

                if mnist_flag == True:
                    sample_size = 16
                    samples = sess.run(output, feed_dict={X: data.test.images[:sample_size]})
                    fig = drawer.plot(samples)
                    plt.savefig(image_dir + '/{}.png'.format(str(epoch_idx+1).zfill(3)), bbox_inches='tight')

            logger.info("Best model idx : " + str(best_model_idx))
            f = open(ckpt_path+'/best_model_idx.txt','w')
            f.write(best_model)
            f.close()

    ##### TEST #####
    if FLAGS.dataset == "Music":
        with tf.Session(config=config) as sess:
            test_total_cost = 0
            if FLAGS.is_train == False:
                if tf.train.get_checkpoint_state(ckpt_dir):
                    f = open(ckpt_dir+'/best_model_idx.txt','r')
                    best_model_idx = f.readline().strip()

            saver.restore(sess, ckpt_path+"-"+str(best_model_idx))
            epoch_idx = eval(best_model_idx)

            test_batch_total = int(data.test.num_examples / FLAGS.batch_size)
            test_logging_step = int(test_batch_total/10);

            for batch_idx in range(test_batch_total):
                batch_xs, batch_idxs = data.test.next_batch_with_idx(FLAGS.batch_size)

                if ((batch_idx+1) % FLAGS.batch_logging_step == 0):
                    #logger.debug('Epoch %.3i, Batch[%.3i/%i], Train loss: %.4E' % (epoch_idx + 1, batch_idx + 1, train_batch_total, cost_val))
                    log_flag = True
                else:
                    log_flag = False

                cost_val = model.inference(sess, batch_xs, epoch_idx, batch_idx, test_batch_total, log_flag)
                #cost_val = sess.run(cost, feed_dict={X: batch_xs})
                test_total_cost += cost_val
                #if ((batch_idx+1) % test_logging_step == 0):
                #if ((batch_idx+1) % FLAGS.batch_logging_step == 0):
                    #logger.debug('Epoch %.3i, Batch[%.3i/%i], Test loss: %.4E' % (epoch_idx + 1, batch_idx + 1, test_batch_total, cost_val))
            logger.debug('Epoch %.3i, Test loss: %.4E' % (epoch_idx + 1, test_total_cost / test_batch_total))

if __name__ == '__main__':
    flags = tf.app.flags

    flags.DEFINE_string("model", "AE", "[RBM, AE, VAE, VAE_GAN, VAE_EBGAN]")
    flags.DEFINE_string("dataset", "MNIST", "Dataset (MNIST,Music) for experiment [MNIST]")

    flags.DEFINE_boolean("is_train", False, "True for training, False for testing [Fasle]")
    flags.DEFINE_boolean("continue_train", None, "True to continue training from saved checkpoint. False for restarting. None for automatic [None]")

    flags.DEFINE_integer("gpu_id", 3, "GPU id [0]")

    flags.DEFINE_string("ae_h_dim_list", "[256]", "List of AE dimensions [256]")
    flags.DEFINE_integer("z_dim", 128, "Dimension of z [128]")

    flags.DEFINE_integer("epoch", 100, "Epoch to train [50]")
    flags.DEFINE_float("learning_rate", 0.01, "Learning rate [0.01]")
    flags.DEFINE_integer("batch_size", 2048, "Batch size [100]")
    flags.DEFINE_integer("batch_logging_step", 10, "Batch size [100]")


    flags.DEFINE_integer("max_to_keep", "11", "maximum number of recent checkpoint files to keep[ckpt]")

    #flags.DEFINE_string("data_dir", "data", 'Directory name to load input data [data]')
    #flags.DEFINE_string("checkpoint_dir", "ckpt", "Directory name to save the checkpoints [checkpoint]")
    #flags.DEFINE_string("log_dir", "log", "Directory name to save the logs [log]")

    FLAGS = flags.FLAGS

    # ckpt path should not contain '[' or ']'
    ae_h_dim_list_replaced = FLAGS.ae_h_dim_list.replace('[','').replace(']','').replace(',','-') 
    model_spec = 'm' + FLAGS.model + '_lr' + str(FLAGS.learning_rate) + '_b' + str(FLAGS.batch_size) + '_h' + ae_h_dim_list_replaced + '_z' + str(FLAGS.z_dim)
        
    if FLAGS.dataset == "MNIST":
        mnist_flag = True
        import matplotlib.pyplot as plt
        from utils import Drawer
        drawer = Drawer()
        image_dir = os.path.join("MNIST/images_visualized", model_spec)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        from tensorflow.examples.tutorials.mnist import input_data
        data = input_data.read_data_sets("MNIST/data/", one_hot=True)
        input_dim = 28*28

    elif FLAGS.dataset == "Music":
        mnist_flag = False
        import data_loader
        data = data_loader.load_music_data()
        input_dim = data.train.dimension

    log_dir = os.path.join(*[FLAGS.dataset, "log", model_spec])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    ckpt_dir = os.path.join(*[FLAGS.dataset, "ckpt", model_spec])
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, "model_ckpt")

    tensorboard_dir = os.path.join(*[FLAGS.dataset, "tensorboard", model_spec])
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    fh = logging.FileHandler(os.path.join(log_dir, 'log'))

    fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    sh.setFormatter(fmt)
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)

    tf.app.run()
