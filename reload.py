import os

import tensorflow as tf
import tensorlayer as tl
from tensorflow.python.framework import graph_util

from model_conv5 import encoder, decoder, discriminator
import import_dataset as datasets
from funcs.preproc import *
import matplotlib.pylab as plt
import numpy as np


# Parameters
BATCH_SIZE = 64
EPOCHS = 800
LR_G = 2e-5
LR = 2e-5
WEIGHT = 0.5
results_path = './Results/Adversarial_Autoencoder'

model_path = "./Results/c1model"
# saver = tf.train.import_meta_graph(model_path + '/-53322.meta')# 加载图结构
# gragh = tf.get_default_graph()# 获取当前图，为了后续训练时恢复变量
# tensor_name_list = [tensor.name for tensor in gragh.as_graph_def().node]# 得到当前图中所有变量的名称


# def freeze_graph(model_folder):
#     # We retrieve our checkpoint fullpath
#     checkpoint = tf.train.get_checkpoint_state(model_folder)
#     input_checkpoint = checkpoint.model_checkpoint_path
#
#     # We precise the file fullname of our freezed graph
#     absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
#     output_graph = absolute_model_folder + "/frozen_model.pb"
#
#     # Before exporting our graph, we need to precise what is our output node
#     # this variables is plural, because you can have multiple output nodes
#     # freeze之前必须明确哪个是输出结点,也就是我们要得到推论结果的结点
#     # 输出结点可以看我们模型的定义
#     # 只有定义了输出结点,freeze才会把得到输出结点所必要的结点都保存下来,或者哪些结点可以丢弃
#     # 所以,output_node_names必须根据不同的网络进行修改
#     output_node_names = 'Decoder/g_conv2/BiasAdd'
#     # We clear the devices, to allow TensorFlow to control on the loading where it wants operations to be calculated
#     clear_devices = True
#
#     # We import the meta graph and retrive a Saver
#     saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
#
#     # We retrieve the protobuf graph definition
#     graph = tf.get_default_graph()
#     input_graph_def = graph.as_graph_def()
#
#     # We start a session and restore the graph weights
#     # 这边已经将训练好的参数加载进来,也即最后保存的模型是有图,并且图里面已经有参数了,所以才叫做是frozen
#     # 相当于将参数已经固化在了图当中
#     with tf.Session() as sess:
#         saver.restore(sess, input_checkpoint)
#
#         # We use a built-in TF helper to export variables to constant
#         output_graph_def = graph_util.convert_variables_to_constants(
#             sess,
#             input_graph_def,
#             output_node_names.split(",")  # We split on comma for convenience
#         )
#
#         # Finally we serialize and dump the output graph to the filesystem
#         with tf.gfile.GFile(output_graph, "wb") as f:
#             f.write(output_graph_def.SerializeToString())
#         print("%d ops in the final graph." % len(output_graph_def.node))
import os
from tensorflow.python.tools.freeze_graph import freeze_graph
from tensorflow.python.ops import variable_scope as vs
def load_graph(frozen_graph_filename):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="",
            # op_dict=None,
            # producer_op_list=None
        )
    return graph
from tensorflow.python.tools import inspect_checkpoint as chkp
def test(z_dim=None, model_name=None):


    f = "./Results/c2/model/"
    # # f = "/data/test/CAAE/Results/Adversarial_Autoencoder/cAdversarial_Autoencoder_WGAN/Saved_models"
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(f+"/-204718.meta",clear_devices=True)  # 加载图结构
    # chkp.print_tensors_in_checkpoint_file(f+"model.ckpt-0", tensor_name='', all_tensors=False,all_tensor_names=True)


    with tf.Session() as sess:

        batch_size = None
        # input_dim = X_train.shape[-1]
        input_dim = 128
        #tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
        graph = tf.get_default_graph()  # 获取当前图，为了后续训练时恢复变量
        # print(graph)
        # tensor_name_list = [tensor.name for tensor in gragh.as_graph_def().node]  # 得到当前图中所有变量的名称
        input = graph.get_tensor_by_name('Input:0')  # 获取输入变量（占位符，由于保存时未定义名称，tf自动赋名称“Placeholder”）
        # saver = tf.train.Saver(tf.all_variables())
        y = graph.get_tensor_by_name('Decoder/g_conv2/BiasAdd:0')  # 获取输出变量

        # En = encoder(input, z_dim,is_train=True)
        # model_de_De = decoder(En,is_train=True)
        # model_de = model_de_De[0]

        # with graph.as_default():

        saver.restore(sess, f + "-204718")
            # graph = tf.get_default_graph()  # 获取当前图，为了后续训练时恢复变量
            # print(graph)
            # tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]  # 得到当前图中所有变量的名称
            # x_input = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim, input_dim, 1],
            #                          name='Input')
            # # x_test_input = tf.placeholder(dtype=tf.float32, shape=[None, input_dim, input_dim, 1],
            # #                          name='Input')
            # x_target = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim, input_dim, 1],
            #                           name='Target')
            # real_distribution = tf.placeholder(dtype=tf.float32, shape=[batch_size, z_dim],
            #                                    name='Real_distribution')
            # decoder_input = tf.placeholder(dtype=tf.float32, shape=[1, z_dim],
            #                                name='Decoder_input')
            #
            # encoder_output = encoder(x_input, z_dim, reuse=tf.AUTO_REUSE, is_train=True)
            #
            #
            # decoder_output, std = decoder(encoder_output, reuse=tf.AUTO_REUSE, is_train=True)
            # encoder_output_z = encoder(decoder_output, z_dim, reuse=tf.AUTO_REUSE, is_train=True)



        all_v = tf.global_variables()
        print("=======================")
        # for v in all_v:
        #     #     # if "moving" in v.name:
        #     print(v.name, v.eval())
        z_dim = 128
        data = datasets.create_datasets(retrain=0, task="aae_wgan_" + str(z_dim), ds_scale=0,
                                        wd="./Data/selscthc")
        X_train = datasets.processNii(data)
        batch_size = BATCH_SIZE
        # input_dim = X_train.shape[-1]
        for batch in tl.iterate.minibatches(inputs=X_train, targets=np.zeros(X_train.shape),
                                            batch_size=batch_size, shuffle=True):
            batch_x, _ = batch
            batch_x = batch_x[:, :, :, np.newaxis]

            # _,new = sess.run([En,model_de],feed_dict={input:batch_x})
            # print("sess.run")
            new= sess.run(y,feed_dict={input:batch_x})

            # ssim = tf.image.ssims(new[])
            # gen = np.array(new[1])
            gen = np.array(new)
            batch_x = batch_x.astype(np.float32)
            for c in range(gen.shape[0]):
                plt.subplot(121)
                plt.imshow(batch_x[c, :, :, 0])
                plt.subplot(122)
                plt.imshow(gen[c, :, :, 0])
                ssim = skimage.measure.compare_ssim(batch_x[c, :, :, 0], gen[c, :, :, 0], data_range=1.0)
                plt.waitforbuttonpress()
                print(ssim)


if  __name__ == '__main__':
    # freeze_graph("D:/python_file/cAAE-master/cAAE-master/Results/c1model/")
    # graph = load_graph("D:/python_file/cAAE-master/cAAE-master/Results/c1model/frozen_model.pb")
    test(128)

