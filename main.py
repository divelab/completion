import os
import time
import argparse
import tensorflow as tf
from network import Generator


dir = '/tempspace/lcai/GAN/Data/data/'

def configure():
    # training
    flags = tf.app.flags
    flags.DEFINE_integer('max_step', 100000, '# of step for training')
    flags.DEFINE_integer('test_interval', 100, '# of interval to test a model')
    flags.DEFINE_integer('save_interval', 1000, '# of interval to save  model')
    flags.DEFINE_integer('summary_interval', 100, '# of step to save summary')
    flags.DEFINE_float('learning_rate', 2e-4, 'learning rate')
    flags.DEFINE_float('trade_off', '0.1', 'trade_off')
    # data
    flags.DEFINE_string('data_dir', dir, 'Name of data directory')
    flags.DEFINE_string('train_data', 'adni_train_all.h5', 'Training data')
    flags.DEFINE_string('valid_data', 'adni_valid.h5', 'Validation data')
    flags.DEFINE_string('test_data', 'adni_test.h5', 'Testing data')
    flags.DEFINE_string('data_type', '3D', '2D data or 3D data')
    flags.DEFINE_integer('batch', 5, 'batch size')
    flags.DEFINE_integer('channel', 1, 'channel size')
    flags.DEFINE_integer('depth', 64, 'depth size')
    flags.DEFINE_integer('height', 64, 'height size')
    flags.DEFINE_integer('width', 64, 'width size')
    # Debug
    flags.DEFINE_string('logdir', './logdir_class_all', 'Log dir')
    flags.DEFINE_string('modeldir', './modeldir_class_all', 'Model dir')
    flags.DEFINE_string('sampledir', './samples/', 'Sample directory')
    flags.DEFINE_string('model_name', 'model', 'Model file name')
    flags.DEFINE_integer('reload_step', 6335, 'Reload step to continue training')
    flags.DEFINE_integer('test_step', 5290, 'Test or predict model at this step')
    flags.DEFINE_integer('random_seed', int(time.time()), 'random seed')
    # network architecture
    flags.DEFINE_integer('network_depth', 5, 'network depth for U-Net')
    flags.DEFINE_integer('class_num', 1, 'output class number')
    flags.DEFINE_integer('start_channel_num', 16,
                         'start number of outputs for the first conv layer')
    flags.DEFINE_string(
        'conv_name', 'conv',
        'Use which conv op in decoder: conv or ipixel_cl')
    flags.DEFINE_string(
        'deconv_name', 'deconv',
        'Use which deconv op in decoder: deconv, pixel_dcl, ipixel_dcl')
    flags.DEFINE_string(
        'action', 'concat',
        'Use how to combine feature maps in pixel_dcl and ipixel_dcl: concat or add')
    # fix bug of flags
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS


def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', dest='option', type=str, default='train',
                        help='actions: train')
    args = parser.parse_args()
    if args.option not in ['train']:
        print('invalid option: ', args.option)
        print("Please input a option: train, test, or predict")
    else:
        model = Generator(tf.Session(), configure())
        getattr(model, args.option)()


if __name__ == '__main__':
    # configure which gpu or cpu to use
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    tf.app.run()
