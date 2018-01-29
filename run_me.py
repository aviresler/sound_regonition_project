import argparse
import os
import tensorflow as tf
from model_sound import Model

"""
This script defines hyper parameters.
"""


def configure():
    flags = tf.app.flags

    # training
    flags.DEFINE_float('initial_learning_rate', 0.01, 'learning rate')
    flags.DEFINE_integer('num_epochs', 1000, 'maximum number of num_epochs')
    flags.DEFINE_float('lr_schedule', [300.0, 600.0], 'after this amount of epochs, the learning rate is divided by 10')
    flags.DEFINE_float('warmup', 10.0, 'for the first warmup epochs, lr is multiplied by 0.1')
    flags.DEFINE_float('momentum', 0.9, 'momentum')
    flags.DEFINE_float('weight_decay', 0.0005, 'weight decay rate')
    flags.DEFINE_integer('k_cross_val', 5, 'number of cross validation sections')
    flags.DEFINE_bool('is_between_class_train', False, 'between class flag')
    flags.DEFINE_string('pretrain_file', '', 'pre-trained model filename')
    flags.DEFINE_integer('fold', 0, 'cross validation fold, training of all epochs will be done according to this split')

    # validation
    flags.DEFINE_integer('num_of_valid_crop', 10, 'number of sections for validation cropping')
    flags.DEFINE_integer('valid_batch_size', 40, 'number of sections for validation cropping')

    # prediction / saving outputs for testing or validation
    flags.DEFINE_string('out_dir', 'output', 'directory for saving outputs')
    flags.DEFINE_integer('save_interval', 50.0, 'number of iterations for saving the model')
    flags.DEFINE_integer('random_seed', 12354, 'random seed')

    # data
    flags.DEFINE_string('data_dir', 'dataset/ESC-50-master/audio/', 'path to data directory')
    flags.DEFINE_string('data_list', 'dataset/ESC-50-master/meta/esc50.csv', 'training data list filename')
    flags.DEFINE_integer('batch_size', 64, 'training batch size')
    flags.DEFINE_integer('input_size', 66650, 'input image width')
    flags.DEFINE_integer('num_classes', 50, 'number of classes')

    # log
    flags.DEFINE_string('modeldir', 'model', 'model directory')
    flags.DEFINE_string('logfile', 'log.txt', 'training log filename')

    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS


def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', dest='option', type=str, default='train',
                        help='actions: train, test, or predict')
    args = parser.parse_args()

    if args.option not in ['train', 'test', 'predict']:
        print('invalid option: ', args.option)
        print("Please input a option: train, test, or predict")
    else:
        # Set up tf session and initialize variables. 
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # sess = tf.Session(config=config)
        sess = tf.Session()
        # Run
        model = Model(sess, configure())
        getattr(model, args.option)()


if __name__ == '__main__':
    tf.app.run()
