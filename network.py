import os
import numpy as np
import tensorflow as tf
from utils.data_reader import H5DataLoader, H53DDataLoader
from utils.img_utils import imsave
from utils import ops


"""
This module builds a standard U-NET for semantic segmentation.
If want VAE using pixelDCL, please visit this code:
https://github.com/HongyangGao/UVAE
"""


class Generator(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.def_params()
        if not os.path.exists(conf.modeldir):
            os.makedirs(conf.modeldir)
        if not os.path.exists(conf.logdir):
            os.makedirs(conf.logdir)
        if not os.path.exists(conf.sampledir):
            os.makedirs(conf.sampledir)
        self.configure_networks()
        self.train_summary = self.config_summary('train')
        self.valid_summary = self.config_summary('valid')

    def def_params(self):
        self.data_format = 'NHWC'
        if self.conf.data_type == '3D':
            self.conv_size = (3, 3, 3)
            self.pool_size = (2, 2, 2)
            self.axis, self.channel_axis = (1, 2, 3), 4
            self.input_shape = [
                self.conf.batch, self.conf.depth, self.conf.height,
                self.conf.width, self.conf.channel]
            self.output_shape = [
                self.conf.batch, self.conf.depth, self.conf.height,
                self.conf.width, self.conf.channel]
        else:
            self.conv_size = (3, 3)
            self.pool_size = (2, 2)
            self.axis, self.channel_axis = (1, 2), 3
            self.input_shape = [
                self.conf.batch, self.conf.height, self.conf.width,
                self.conf.channel]
            self.output_shape = [
                self.conf.batch, self.conf.height, self.conf.width]

    def configure_networks(self):
        self.build_network()
        trainable_vars = tf.trainable_variables()
        self.g_vars = [var for var in trainable_vars if var.name.startswith('g')]
        self.d_vars = [var for var in trainable_vars if var.name.startswith('d')]
        self.g_train = tf.train.AdamOptimizer(1e-3, beta1 = 0.5, beta2 =0.999).minimize(self.g_loss_total, var_list = self.g_vars)
        self.d_train = tf.train.AdamOptimizer(2e-4, beta1 = 0.5, beta2 =0.999).minimize(self.d_loss_total, var_list = self.d_vars)
        tf.set_random_seed(self.conf.random_seed)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)

    def build_network(self):
        self.catgory = tf.placeholder(
            tf.int32, [self.conf.batch,1], name = 'catgory')
        self.inputs = tf.placeholder(
            tf.float32, self.input_shape, name='inputs')
        self.labels = tf.placeholder(
            tf.float32, self.output_shape, name='labels')
        self.predictions = self.inference(self.inputs)
        self.d, self.d_logits, self.c, self.c_logits = self.discriminator(self.inputs, self.labels, reuse = False)
        self.d_, self.d_logits_, self.c_, self.c_logits_ = self.discriminator(self.inputs, self.predictions, reuse = True)
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits = self.d_logits, labels = tf.ones_like(self.d)) )
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits = self.d_logits_, labels = tf.zeros_like(self.d_)) )
        self.c_loss_real = tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(labels = self.catgory, logits = self.c_logits))
        self.c_loss_fake = tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(labels = self.catgory, logits = self.c_logits_))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(self.d_), logits = self.d_logits_))
        self.cal_loss()

    def cal_loss(self):
        self.completion = tf.losses.mean_squared_error(self.labels, self.predictions)
        self.g_loss_total =  self.completion + self.conf.trade_off*(self.g_loss + self.c_loss_fake)
        self.d_loss_total = self.d_loss_fake + self.d_loss_real + self.c_loss_real + self.c_loss_fake



    def config_summary(self, name):
        summarys = []
        summarys.append(tf.summary.scalar(name+'/d_loss_total', self.d_loss_total))
        summarys.append(tf.summary.scalar(name+'/g_loss_total', self.g_loss_total))
        summarys.append(tf.summary.scalar(name+'/d_loss_real', self.d_loss_real))
        summarys.append(tf.summary.scalar(name+'/d_loss_fake', self.d_loss_fake))
        summarys.append(tf.summary.scalar(name+'/c_loss_real', self.c_loss_real))
        summarys.append(tf.summary.scalar(name+'/c_loss_fake', self.c_loss_fake))
        summarys.append(tf.summary.scalar(name+'/completion_loss', self.completion))
        summarys.append(tf.summary.scalar(name+'/g_loss', self.g_loss))
        summary = tf.summary.merge(summarys)
        return summary

    def discriminator(self, mri, pet, reuse = False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            inputs = tf.concat(
                [mri, pet], self.channel_axis, name='/concat')
            print('input',' ',inputs.shape)
            conv1 = ops.conv(inputs, 16, (3, 3, 3), '/conv1', self.conf.data_type)
            conv1 = ops.batch_norm(conv1, '/batch1',ac_fn = ops.leaky_relu)
            print('conv1 ',conv1.shape)
            conv1 = tf.nn.dropout(conv1, 0.5, name='drop1')
            conv2 = ops.conv(conv1, 32, (3, 3, 3), '/conv2', self.conf.data_type,2)
            conv2 = ops.batch_norm(conv2, '/batch2',ac_fn = ops.leaky_relu)
            print('conv2 ',conv2.shape)
            conv2 = tf.nn.dropout(conv2, 0.5, name = 'drop2')
            conv3 = ops.conv(conv2, 64, (3, 3, 3), '/conv3', self.conf.data_type)
            conv3 = ops.batch_norm(conv3, '/batch3',ac_fn = ops.leaky_relu)
            print('conv3 ',conv3.shape)
            conv3 = tf.nn.dropout(conv3, 0.5, name = 'drop3')
            conv4 = ops.conv(conv3, 128, (3, 3, 3), '/conv4', self.conf.data_type,2)
            conv4 = ops.batch_norm(conv4, '/batch4',ac_fn = ops.leaky_relu)
            conv4 = tf.nn.dropout(conv4, 0.5, name = 'drop4')
            print('conv4 ',conv4.shape)
            flatten = tf.contrib.layers.flatten(conv4)
            logits = tf.contrib.layers.fully_connected(flatten, 5, activation_fn=None, scope = '/fully')
            print('flatten ',flatten.shape)
            print('logits ',logits.shape)
            d = tf.nn.sigmoid(logits)
            return d[:,0], logits[:,0], d[:,1:5], logits[:,1:5]

    def inference(self, inputs):
        with tf.variable_scope('generator') as scope:
            outputs = inputs
            down_outputs = []
            for layer_index in range(self.conf.network_depth-1):
                is_first = True if not layer_index else False
                name = 'down%s' % layer_index
                outputs = self.build_down_block(
                    outputs, name, down_outputs, is_first)
            outputs = self.build_bottom_block(outputs, 'bottom')
            for layer_index in range(self.conf.network_depth-2, -1, -1):
                is_final = True if layer_index == 0 else False
                name = 'up%s' % layer_index
                down_inputs = down_outputs[layer_index]
                outputs = self.build_up_block(
                    outputs, down_inputs, name, is_final)
            return outputs

    def build_down_block(self, inputs, name, down_outputs, first=False):
        out_num = self.conf.start_channel_num if first else 2 * \
            inputs.shape[self.channel_axis].value
        conv1 = ops.conv(inputs, out_num, self.conv_size,
                         name+'/conv1', self.conf.data_type)
        conv1 = ops.batch_norm(conv1, name+'/batch1')
        conv2 = ops.conv(conv1, out_num, self.conv_size,
                         name+'/conv2', self.conf.data_type,2)
        conv2 = ops.batch_norm(conv2, name+'/batch2')
        down_outputs.append(conv1)
        return conv2

    def build_bottom_block(self, inputs, name):
        out_num = inputs.shape[self.channel_axis].value
        conv1 = ops.conv(
            inputs, 2*out_num, self.conv_size, name+'/conv1',
            self.conf.data_type)
        conv1 = ops.batch_norm(conv1, name+'/batch1')
        conv2 = ops.conv(
            conv1, out_num, self.conv_size, name+'/conv2', self.conf.data_type)
        conv2 = ops.batch_norm(conv2, name+'/batch2')
        return conv2

    def build_up_block(self, inputs, down_inputs, name, final=False):
        out_num = inputs.shape[self.channel_axis].value
        conv1 = self.deconv_func()(
            inputs, out_num, self.conv_size, name+'/conv1',
            self.conf.data_type, action=self.conf.action)
        conv1 = ops.batch_norm(conv1, name+'/batch1')
        conv1 = tf.concat(
            [conv1, down_inputs], self.channel_axis, name=name+'/concat')
        conv2 = self.conv_func()(
            conv1, out_num, self.conv_size, name+'/conv2', self.conf.data_type)
        conv2 = ops.batch_norm(conv2, name+'/batch2')
        out_num = self.conf.class_num if final else out_num/2
        if final:
            conv3 = ops.conv(conv2, out_num, self.conv_size, name+'/conv3', self.conf.data_type)
        else:
            conv3 = ops.conv(conv2, out_num, self.conv_size, name+'/conv3', self.conf.data_type)
            conv3 = ops.batch_norm(conv3, name+'/batch3')
        return conv3

    def deconv_func(self):
        return getattr(ops, self.conf.deconv_name)

    def conv_func(self):
        return getattr(ops, self.conf.conv_name)

    def save_summary(self, summary, step):
        print('---->summarizing', step)
        self.writer.add_summary(summary, step)

    def train(self):
        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)
        train_reader = H5DataLoader(
            self.conf.data_dir+self.conf.train_data)
        valid_reader = H5DataLoader(
            self.conf.data_dir+self.conf.valid_data)
        iteration = train_reader.iter + self.conf.reload_step
        pre_iter = iteration
        epoch_num = 0
        while iteration < self.conf.max_step:
            if pre_iter != iteration:
                pre_iter = iteration
                inputs, labels, catgory = valid_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: inputs,
                             self.labels: labels,
                             self.catgory: catgory}
                loss, summary = self.sess.run(
                    [self.d_loss_total, self.valid_summary], feed_dict=feed_dict)
                self.save_summary(summary, iteration)
                print('----testing d loss', loss)
                loss, summary = self.sess.run(
                    [self.g_loss_total, self.valid_summary], feed_dict=feed_dict)
                self.save_summary(summary, iteration)
                self.save(iteration)
                print('----testing g loss', loss)
            elif epoch_num % self.conf.summary_interval == 0:
                inputs, labels, catgory = train_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: inputs,
                             self.labels: labels,
                             self.catgory: catgory}
                loss, _, summary = self.sess.run(
                    [self.d_loss_total, self.d_train, self.train_summary], feed_dict=feed_dict)
                self.save_summary(summary, epoch_num+self.conf.reload_step)
                loss, _, summary = self.sess.run(
                    [self.g_loss_total, self.g_train, self.train_summary],
                    feed_dict=feed_dict)
                self.save_summary(summary, epoch_num+self.conf.reload_step)
            else:
                inputs, labels, catgory = train_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: inputs,
                             self.labels: labels,
                             self.catgory: catgory}
                loss, _, summary = self.sess.run(
                    [self.d_loss_total, self.d_train, self.train_summary], feed_dict=feed_dict)
                print('----training d loss', loss)
                loss, _, summary = self.sess.run(
                    [self.g_loss_total, self.g_train, self.train_summary], feed_dict=feed_dict)
                print('----training g loss', loss)
            iteration = train_reader.iter + self.conf.reload_step
            epoch_num += 1
    def test(self):
        print('---->predicting ', self.conf.test_step)
        if self.conf.test_step > 0:
            self.reload(self.conf.test_step)
        else:
            print("please set a reasonable test_step")
            return
        data_mri, label = load_data('test_data.mat')
        predictions = []
        for i in range(int(data_mri.shape[0]/self.conf.batch)):
            print(i ," in ", int(data_mri.shape[0]/self.conf.batch))
            inputs = np.reshape(data_mri[i*self.conf.batch:i*self.conf.batch+self.conf.batch], self.input_shape)
            annotations = np.reshape(data_mri[i*self.conf.batch:i*self.conf.batch+self.conf.batch], self.input_shape)
            feed_dict = {self.inputs: inputs, self.labels: annotations}
            predictions.append(self.sess.run(
                self.predictions, feed_dict=feed_dict))
        predictions = np.concatenate(predictions,axis=0)
        np.savez("samples_unet"+str(self.conf.test_step),predictions)

    def save(self, step):
        print('---->saving', step)
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        model_path = checkpoint_path+'-'+str(step)
        if not os.path.exists(model_path+'.meta'):
            print('------- no such checkpoint', model_path)
            return
        self.saver.restore(self.sess, model_path)



