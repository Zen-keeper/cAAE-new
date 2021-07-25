# python -m pdb aae_wgan.py --load 0 --comment "aae wgan" --model_name "None" --step "0"
# python -m pdb aae_wgan.py --load 1 --comment "2018-02-22 12:40:34.603793_35_Adversarial_Autoencoder_WGAN retrain" --model_name "2018-02-22 12:40:34.603793_35_Adversarial_Autoencoder_WGAN" --step "-4933"


import os
os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU_NAME'
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from model_conv4 import encoder, decoder, discriminator
import import_dataset as datasets
from funcs.preproc import *
from sklearn.model_selection import KFold
# Parameters
BATCH_SIZE = 64
EPOCHS = 300
LR_G = 2e-4
LR = 2e-4
WEIGHT = 0.5
results_path = './Results/Adversarial_Autoencoder'
# class LearningRate(object):
#     def __init__(self):
#         self.init_learningrate = LR
#         self.decay_rate = 0.5
#         # self.global_steps = 1000
#         self.decay_steps = 1000000
#         self.global_ = tf.Variable(tf.constant(0))
# 		#每decay_steps对学习率进行衰减
#         self.learning_rate1 = tf.train.exponential_decay(
#             self.init_learningrate, self.global_, self.decay_steps,
#             self.decay_rate, staircase = True, name='learning_rate1')
#             #每步都对学习率进行衰减
#         self.learning_rate2 = tf.train.exponential_decay(
#             self.init_learningrate, self.global_, self.decay_steps,
#             self.decay_rate, staircase=False, name='learning_rate2')
# T = []
# F = []
#
# def main():
#     with tf.Session() as sess:
#         a = LearningRate()
#         for i in range(a.global_steps):
#             F_c = sess.run(a.learning_rate2,feed_dict={a.global_:i})
#             F.append(F_c)
#             T_c = sess.run(a.learning_rate1,feed_dict={a.global_: i})
#             T.append(T_c)
#     plt.figure(1)
#     plt.plot(range(a.global_steps),T,'r-')
#     plt.plot(range(a.global_steps), F, 'b-')


def form_results():
    """mirrors.hoc.ccshu.net/jsc/gpu:tensorlayer_
    Forms folders for each run to store the tensorboard files, saved models and the log files.
    :return: three string pointing to tensorboard, saved models and log paths respectively.
    """
    results_path = './Results/Adversarial_Autoencoder'
    folder_name = "/cAdversarial_Autoencoder_WGAN"
    tensorboard_path = results_path + folder_name + '/Tensorboard'
    saved_model_path = results_path + folder_name + '/Saved_models/'
    log_path = results_path + folder_name + '/log'
    if not os.path.exists(results_path + folder_name):
        os.mkdir(results_path + folder_name)
        os.mkdir(tensorboard_path)
        os.mkdir(saved_model_path)
        os.mkdir(log_path)
    return tensorboard_path, saved_model_path, log_path, folder_name

def train(z_dim=None, model_name=None):
    """
    Used to train the autoencoder by passing in the necessary inputs.
    :param train_model: True -> Train the model, False -> Load the latest trained model and show the image grid.
    :return: does not return anything
    """

    if not os.path.exists(results_path + "/ADtest_img"):
        os.mkdir(results_path + "/ADtest_img")
    z_dim = int(z_dim)


    batch_size = None
    # input_dim = X_train.shape[-1]
    input_dim = 128
    tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES

    with tf.device("/gpu:0"):
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        # log_device_placement = True: 是否打印设备分配日志
        # allow_soft_placement = True: 如果你指定的设备不存在，允许TF自动分配设备
        x_input = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim, input_dim, 1],
                                 name='Input')
        # x_test_input = tf.placeholder(dtype=tf.float32, shape=[None, input_dim, input_dim, 1],
        #                          name='Input')
        x_target = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim, input_dim, 1],
                                  name='Target')
        real_distribution = tf.placeholder(dtype=tf.float32, shape=[batch_size, z_dim],
                                           name='Real_distribution')
        decoder_input = tf.placeholder(dtype=tf.float32, shape=[1, z_dim],
                                       name='Decoder_input')
        # learning_rate = tf.Variable(LR)

        encoder_output = encoder(x_input, z_dim, reuse=False, is_train=True)
        encoder_output_test = encoder(x_input, z_dim, reuse=True, is_train=False)
        # encoder_z_in_test = encoder(x_test_input, z_dim, reuse=True, is_train=False)
        d_fake, d_fake_logits = discriminator(encoder_output, reuse=False)
        d_real, d_real_logits = discriminator(real_distribution, reuse=True)

        d_fake_test, d_fake_logits_test = discriminator(encoder_output, reuse=True)
        d_real_test, d_real_logits_test = discriminator(real_distribution, reuse=True)

        decoder_output, std = decoder(encoder_output, reuse=False, is_train=True)
        encoder_output_z = encoder(decoder_output, z_dim, reuse=True, is_train=False)
        decoder_output_test, std_ = decoder(encoder_output, reuse=True, is_train=False)
        encoder_output_z_test = encoder(decoder_output_test, z_dim, reuse=True, is_train=False)
        #测试集解码
        # decoder_output_in_test = decoder(encoder_z_in_test, reuse=True, is_train=False)
        # decoder_image = decoder(decoder_input, reuse=True, is_train=False)
        # Autoencoder loss
        # summed = tf.reduce_mean(tf.square(decoder_output-x_target),[1,2,3])
        summed = tf.reduce_sum(tf.square(decoder_output - x_target), [1, 2, 3])
        # sqrt_summed = summed
        sqrt_summed = tf.sqrt(summed + 1e-8)
        autoencoder_loss = tf.reduce_mean(sqrt_summed)

        summed_test = tf.reduce_sum(tf.square(decoder_output_test - x_target), [1, 2, 3])#mse
        # sqrt_summed_test = summed_test
        sqrt_summed_test = tf.sqrt(summed_test + 1e-8)
        autoencoder_loss_test = tf.reduce_mean(sqrt_summed_test)

        # l2 loss of z
        enc = tf.reduce_sum(tf.square(encoder_output - encoder_output_z), [1])
        encoder_l2loss = tf.reduce_mean(enc)
        enc_test = tf.reduce_sum(tf.square(encoder_output_test - encoder_output_z_test), [1])#mse z
        encoder_l2loss_test = tf.reduce_mean(enc_test)

        dc_loss = tf.reduce_mean(d_real_logits - d_fake_logits)
        dc_loss_test = tf.reduce_mean(d_real_logits_test - d_fake_logits_test)
        ssim = tf.image.ssim(x_input, decoder_output, max_val=1)
        ssim = tf.reduce_mean(ssim)
        ssim_test = tf.reduce_mean(tf.image.ssim(x_input, decoder_output_test, max_val=1))

        with tf.name_scope("Gradient_penalty"):
            eta = tf.placeholder(tf.float32, shape=[batch_size, 1], name="Eta")
            interp = eta * real_distribution + (1 - eta) * encoder_output
            _, c_interp = discriminator(interp, reuse=True)

            # taking the zeroth and only element because tf.gradients returns a list
            c_grads = tf.gradients(c_interp, interp)[0]

            # L2 norm, reshaping to [batch_size]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(c_grads), axis=[1]))
            tf.summary.histogram("Critic gradient L2 norm", slopes)

            grad_penalty = tf.reduce_mean((slopes - 1) ** 2)
            lambd = 10.0
            dc_loss += lambd * grad_penalty

        # Generator loss
        generator_loss = tf.reduce_mean(
           tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake_logits))
        # generator_loss = tf.reduce_mean(d_fake_logits)
        generator_loss_test = tf.reduce_mean(d_fake_logits_test)

        all_variables = tf.trainable_variables()
        dc_var = tl.layers.get_variables_with_name('Discriminator', True, True)
        en_var = tl.layers.get_variables_with_name('Encoder', True, True)
        decoder_var = tl.layers.get_variables_with_name('Decoder', True, True)
        #print en_var
        # dc_var = [var for var in all_variables if 'dc' in var.name]
        # en_var = [var for var in all_variables if 'encoder' in var.name]
        var_grad_autoencoder = tf.gradients(autoencoder_loss, all_variables)[0]
        var_grad_discriminator = tf.gradients(dc_loss, dc_var)[0]
        var_grad_generator = tf.gradients(generator_loss, en_var)[0]

        # ops = tf.get_default_graph().get_operations()
        # update_ops = [x for x in ops if ("AssignMovingAvg" in x.name and x.type == "AssignSub")]
        # # update_ops = [x for x in ops if ("bn" in x.name)]
        # tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_ops)

        # Optimizers
        with tf.device("/gpu:0"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                global_step_G = tf.Variable(0, trainable=False)
                global_step_D = tf.Variable(0, trainable=False)
                starter_learning_rate = LR
                # learning_rate_G = tf.train.exponential_decay(starter_learning_rate, global_step_G, 55000, 0.9, staircase=True)
                # learning_rate_D = tf.train.exponential_decay(starter_learning_rate, global_step_D, 110000, 0.9, staircase=True)
                # learning_rate_G = tf.train.piecewise_constant(global_step_G, [200000,400000], [LR,LR/2,LR/10])
                # learning_rate_D = tf.train.piecewise_constant(global_step_D, [200000,400000], [LR,LR/2,LR/10])
                learning_rate_G = LR_G
                learning_rate_D = LR
                autoencoderl2_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_G,
                                                                 beta1=0.5, beta2=0.9).minimize(
                    dc_loss + 10.0* autoencoder_loss + 0.5 *encoder_l2loss,var_list=[en_var,decoder_var])#,global_step=global_step
                autoencoder_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_G,
                                                               beta1=0.5, beta2=0.9).minimize(autoencoder_loss)

                discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_D,
                                                                 beta1=0.5, beta2=0.9).minimize(dc_loss, var_list=dc_var)

                generator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_G,
                                                             beta1=0.5, beta2=0.9).minimize(generator_loss, var_list=en_var)

                tl.layers.initialize_global_variables(sess)
        # Reshape immages to display them
        input_images = tf.reshape(x_input, [-1, input_dim, input_dim, 1])
        # generated_images =tf.scalar_mul(tf.add(tf.reshape(decoder_output, [-1, input_dim, input_dim, 1]),tf.constant(1.0)),tf.constant(255.0/2))
        generated_images = tf.reshape(decoder_output, [-1, input_dim, input_dim, 1])
        tensorboard_path, saved_model_path, log_path, folder_name = form_results()
        # bp()
        writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=sess.graph)
        # Tensorboard visualization
        tf.summary.scalar(name='Autoencoder Loss', tensor=autoencoder_loss)
        tf.summary.scalar(name='Autoencoder Test Loss', tensor=autoencoder_loss_test)
        tf.summary.scalar(name='Discriminator Loss', tensor=dc_loss)
        tf.summary.scalar(name='Generator Loss', tensor=generator_loss)
        tf.summary.scalar(name='Autoencoder z Loss', tensor=encoder_l2loss)
        tf.summary.scalar(name='SSIM', tensor=ssim)
        tf.summary.scalar(name='SSIMtest', tensor=ssim_test)
        tf.summary.histogram(name='Encoder Distribution', values=encoder_output)
        tf.summary.histogram(name='Decoder Distribution', values=decoder_output)
        tf.summary.histogram(name='Real Distribution', values=real_distribution)
        tf.summary.histogram(name='Gradient AE', values=var_grad_autoencoder)
        tf.summary.histogram(name='Gradient D', values=var_grad_discriminator)
        tf.summary.histogram(name='Gradient G', values=var_grad_generator)
        tf.summary.image(name='Input Images', tensor=input_images, max_outputs=10)
        tf.summary.image(name='Generated Images', tensor=generated_images, max_outputs=10)
        summary_op = tf.summary.merge_all()


        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        # bn_moving_vars += [g for g in g_list if 'beta' in g.name]
        # bn_moving_vars += [g for g in g_list if 'bias' in g.name]
        var_list += bn_moving_vars

        saver = tf.train.Saver(var_list=var_list, max_to_keep=5)
        # saver = tf.train.Saver()
    # Saving the model

    step = 0
    # with tf.Session() as sess:
    with open(log_path + '/log.txt', 'a') as log:
        log.write("input_dim: {}\n".format(input_dim))
        log.write("z_dim: {}\n".format(z_dim))
        log.write("batch_size: {}\n".format(batch_size))
        log.write("\n")
    init = tf.global_variables_initializer()
    init_op = tf.initialize_local_variables()
    sess.run(init_op)
    sess.run(init)

    tl.layers.initialize_global_variables(sess)

    # # 创建表格
    # f = xlwt.Workbook()
    # sheet_nc = f.add_sheet('ssim_test', cell_overwrite_ok=True)
    # sheet_xuanwu = f.add_sheet('ssim_xuanwu', cell_overwrite_ok=True)
    # row = []
    # row.append("name")
    # for i in range(100):
    #     row.append(str(i))
    # for i in range(0, len(row)):
    #     sheet_nc.write(0, i, row[i])
    #     sheet_xuanwu.write(0, i, row[i])
    data = datasets.create_datasets(retrain=0, task="aae_wgan_" + str(z_dim), ds_scale=0,
                                    wd="Data/AD_NC/dataNC")  # X_train, y_train
    X_train = datasets.processNii(data)
    for epoch in range(EPOCHS):
        #K折读取训练数据
        # kf = KFold(n_splits=10)
        # k_num=0
        # for train_index, tests_index in kf.split(data):
        # k_num = k_num+1
        # X_train, X_test = data[train_index], data[tests_index]
        b = 0
        #切片处理
        # X_train = datasets.processNii(X_train)
        # X_test = datasets.processNii(X_test)
        for batch in tl.iterate.minibatches(inputs=X_train, targets=np.zeros(X_train.shape),
                                            batch_size=BATCH_SIZE, shuffle=True):
            z_real_dist = np.random.normal(0, 1, (BATCH_SIZE, z_dim)) * 1.
            z_real_dist = z_real_dist.astype("float32")

            batch_x, _ = batch
            batch_x = batch_x[:, :, :, np.newaxis]

            #################################
            #train autoencoder=====generator
            #################################
            eta1 = np.random.rand(BATCH_SIZE, 1)  # sampling from uniform distribution
            eta1 = eta1.astype("float32")#训练的参数有encoder\decoder，（通过1*L2损失、0.5*z损失、1*辨别器的鉴别误差）
            sess.run(autoencoderl2_optimizer, feed_dict={x_input: batch_x, x_target: batch_x,real_distribution: z_real_dist,eta: eta1})

            # ################################
            # train discrimator
            # #################################
            if epoch < 100:
                # sess.run(autoencoder_optimizer, feed_dict={x_input: batch_x, x_target: batch_x})
                for t in range(1):#10
                    for _ in range(2):#20
                        eta1 = np.random.rand(BATCH_SIZE, 1)  # sampling from uniform distribution
                        eta1 = eta1.astype("float32")
                        sess.run(discriminator_optimizer,
                                 feed_dict={x_input: batch_x, x_target: batch_x,
                                            real_distribution: z_real_dist, eta: eta1})
                        # global_step +=1

            else:
                # sess.run(autoencoderl2_optimizer, feed_dict={x_input: batch_x, x_target: batch_x})
                for _ in range(1):
                    eta1 = np.random.rand(BATCH_SIZE, 1)  # sampling from uniform distribution
                    eta1 = eta1.astype("float32")
                    sess.run(discriminator_optimizer,
                             feed_dict={x_input: batch_x, x_target: batch_x,
                                        real_distribution: z_real_dist, eta: eta1})
            sess.run(generator_optimizer, feed_dict={x_input: batch_x, x_target: batch_x})
            if b % 20 == 0:
                a_loss, e_loss, d_loss, g_loss, a_grad, d_grad, g_grad, en_output, d_real_logits_, d_fake_logits_, de_output, ssim_value, summary = sess.run(
                    [autoencoder_loss, encoder_l2loss, dc_loss, generator_loss, var_grad_autoencoder,
                     var_grad_discriminator,
                     var_grad_generator, encoder_output, d_real_logits, d_fake_logits, decoder_output, ssim, summary_op],
                    feed_dict={x_input: batch_x, x_target: batch_x,
                               real_distribution: z_real_dist, eta: eta1})

                print(model_name)
                saver.save(sess, save_path=saved_model_path, global_step=step)
                writer.add_summary(summary, global_step=step)

                print("Epoch: {}, iteration: {}".format(epoch, b))
                print("Autoencoder Loss: {}".format(a_loss))
                print("Autoencoder enc Loss: {}".format(e_loss))
                print("Discriminator Loss: {}".format(d_loss))
                print("Generator Loss: {}".format(g_loss))
                print("SSIM: {}".format(ssim_value))
                with open(log_path + '/log.txt', 'a') as log:
                    log.write("Epoch: {}, iteration: {}\n".format(epoch, b))
                    log.write("Autoencoder Loss: {}\n".format(a_loss))
                    log.write("Autoencoder enc Loss: {}\n".format(e_loss))
                    log.write("Discriminator Loss: {}\n".format(d_loss))
                    log.write("Generator Loss: {}\n".format(g_loss))

                # print('Learning rate of G: %f' % (sess.run(autoencoderl2_optimizer._lr)))
                # print('Learning rate of D: %f' % (sess.run(discriminator_optimizer._lr)))
            b += 1
            step += 1


        # if epoch%2==0:
        #     b=0
        #     for (name, batch) in datasets.create_test("./Data/xuanwuhc"):
        #         # sheet_nc.write(b+1, 0, name)
        #         # if (not os.path.exists(results_path + "/ADNI_NC_test_img/")):
        #         #     os.mkdir(results_path + "/ADNI_NC_test_img")
        #         # if (not os.path.exists(results_path + "/ADNI_NC_test_img/k" +str(k_num)+"_"+ str(b)+ "_"  + name)):
        #         #     os.mkdir(results_path + "/ADNI_NC_test_img/k" +str(k_num)+"_"+ str(b)+ "_"  + name)
        #         batch_x = batch
        #         batch_x = batch_x[:, :, :, np.newaxis]
        #         batch_size = batch_x.shape[0]
        #         z_real_dist = np.random.normal(0, 1, (batch_size, z_dim)) * 1.
        #         z_real_dist = z_real_dist.astype("float32")
        #
        #         eta1 = np.random.rand(batch_x.shape[0], 1)
        #         if b % 2 == 0:
        #             gen_nc, ssim_test_value = sess.run(
        #                 [decoder_output_test, ssim_test],
        #                 feed_dict={x_input: batch_x, x_target: batch_x,
        #                            real_distribution: z_real_dist, eta: eta1})
        #         # nii_ori = []
        #         # nii_gen = []
        #         # for i in range(gen_nc.shape[0]):
        #         #     img = np.concatenate((batch_x[i, :, :, 0], gen_nc[i, :, :, 0]), axis=1)
        #         #     # if not os.path.exists(results_path + "/ADNI_nc_test_img"):
        #         #     #     os.mkdir(results_path + "/ADNI_nc_test_img")
        #         #     nii_ori.append(np.flipud(batch_x[i, :, :, 0]))
        #         #     nii_gen.append(np.flipud(gen_nc[i, :, :, 0]))
        #         #     plt.imsave(results_path + "/ADNI_NC_test_img/k" +str(k_num)+"_"+ str(b) + "_" + name+"/slice_"+str(i)+".png",img)
        #         # nii_ori = sitk.GetImageFromArray(np.array(nii_ori))
        #         # nii_gen = sitk.GetImageFromArray(np.array(nii_gen))
        #         # sitk.WriteImage(nii_ori, results_path + "/ADNI_NC_test_img/k" +str(k_num)+"_"+ str(b)+ "_"  + name+"/"+"_ori_" +name+".nii")
        #         # sitk.WriteImage(nii_gen, results_path + "/ADNI_NC_test_img/k" +str(k_num)+"_"+ str(b)+ "_" + name+"/" + "_gen_" +name+".nii")
        #         # del nii_ori
        #         # del nii_gen
        #             print("v_Epoch: {}, iteration: {}".format(epoch, b))
        #             print("v_SSIM: {}".format(ssim_test_value))
        #
        #
        #             with open(log_path + '/log.txt', 'a') as log:
        #                 log.write("========test in adni cn ========")
        #                 log.write("test_Epoch: {}, iteration: {}\n".format(epoch, b))
        #                 log.write("test_SSIM: {}\n".format(ssim_test_value))
        #         b = b + 1
        #         # f.save("./ssim_cn.xls")
        #     del X_train, X_test
        #
        #
        # if  epoch%2==0:
        #     b = 0  # 对象序号
        #     for (name, batch) in datasets.create_test("./Data/xuanwuhc"):
        #         if (not os.path.exists(results_path + "/xuanwu_test_img/" + str(b) + "_" + name)):
        #             os.mkdir(results_path + "/xuanwu_test_img/" + str(b) + "_" + name)
        #         batch_x = batch
        #         batch_x = batch_x[:, :, :, np.newaxis]
        #         batch_size = batch_x.shape[0]
        #         z_real_dist = np.random.normal(0, 1, (batch_size, z_dim)) * 1.
        #         z_real_dist = z_real_dist.astype("float32")
        #
        #         eta1 = np.random.rand(batch_x.shape[0], 1)
        #         # if b % 10 == 0:
        #         gen_test, ssim_test_value = sess.run(
        #             [decoder_output_test, ssim_test],
        #             feed_dict={x_input: batch_x, x_target: batch_x,
        #                        real_distribution: z_real_dist, eta: eta1})
        #         nii_ori = []
        #         nii_gen = []
        #         for i in range(gen_test.shape[0]):
        #             # residual = np.abs(np.subtract(gen_test[i, :, :, 0], batch_x[i, :, :, 0]))
        #             img = np.concatenate((batch_x[i, :, :, 0], gen_test[i, :, :, 0]), axis=1)
        #             nii_ori.append(np.flipud(batch_x[i, :, :, 0]))
        #             nii_gen.append(np.flipud(gen_test[i, :, :, 0]))
        #             plt.imsave(results_path + "/xuanwu_test_img/" + str(b) + "_" + name + "/slice_" + str(i) + ".png",
        #                        img)
        #         nii_ori = sitk.GetImageFromArray(np.array(nii_ori))
        #         nii_gen = sitk.GetImageFromArray(np.array(nii_gen))
        #         sitk.WriteImage(nii_ori, results_path + "/xuanwu_test_img/" + str(b) + "_" + name + "/" +"_ori_" + name + ".nii")
        #         sitk.WriteImage(nii_gen, results_path + "/xuanwu_test_img/" + str(b) + "_" + name + "/"+"_gen_" + name + ".nii")
        #         del nii_ori
        #         del nii_gen
        #         print("=======test in xuanwu hc========")
        #         # print("test_Generator Loss: {}".format(g_loss))
        #         print("test_SSIM: {}".format(ssim_test_value))
        #         sheet_xuanwu.write(b+1,0,name)
        #         sheet_xuanwu.write(b+1,1,str(ssim_test_value))
        #
        #         b = b + 1
        #         with open(log_path + '/log.txt', 'a') as log:
        #             log.write("========test in xuanwu ========")
        #             log.write("test_Epoch: {}, iteration: {}\n".format(epoch, b))
        #             log.write("test_SSIM: {}\n".format(ssim_test_value))
        # # ==================test AD in ADNI============================================
        # if epoch%2==0:
        #     b = 0#对象序号
        #     for (name,batch) in datasets.create_test("./Data/selectAD"):
        #         if(not os.path.exists(results_path + "/ADtest_img/"+str(b)+"_"+name)):
        #             os.mkdir(results_path + "/ADtest_img/"+str(b)+"_"+name)
        #         batch_x= batch
        #         batch_size = batch_x.shape[0]
        #         z_real_dist = np.random.normal(0, 1, (batch_size, z_dim)) * 1.
        #         z_real_dist = z_real_dist.astype("float32")
        #         batch_x = batch_x[:, :, :, np.newaxis]
        #         eta1 = np.random.rand(batch_size, 1)
        #         # if b % 10 == 0:
        #         gen_test, a_loss, e_loss, d_loss, g_loss, ssim_test_value = sess.run(
        #             [decoder_output_test,autoencoder_loss_test, encoder_l2loss_test, dc_loss_test, generator_loss_test, ssim_test],
        #             feed_dict={x_input: batch_x, x_target: batch_x,
        #                        real_distribution: z_real_dist, eta: eta1})
        #         nii_ori = []
        #         nii_gen = []
        #         for i in range(gen_test.shape[0]):
        #             # residual = np.abs(np.subtract(gen_test[i, :, :, 0], batch_x[i, :, :, 0]))
        #             img = np.concatenate((batch_x[i, :, :, 0], gen_test[i, :, :, 0]), axis=1)
        #             nii_ori.append(np.flipud(batch_x[i, :, :, 0]))
        #             nii_gen.append(np.flipud(gen_test[i, :, :, 0]))
        #             plt.imsave(results_path + "/ADtest_img/"+str(b)+"_"+name+"/slice_"+str(i)+".png",img)
        #         nii_ori = sitk.GetImageFromArray(np.array(nii_ori))
        #         nii_gen = sitk.GetImageFromArray(np.array(nii_gen))
        #         sitk.WriteImage(nii_ori, results_path + "/ADtest_img/" + str(b) + "_" + name + "/" + "_ori_" + name + ".nii")
        #         sitk.WriteImage(nii_gen, results_path + "/ADtest_img/" + str(b) + "_" + name + "/" + "_gen_" + name + ".nii")
        #         del nii_ori
        #         del nii_gen
        #
        #
        #         print("=======test in ad========")
        #         print("test_SSIM: {}".format(ssim_test_value))
        #
        #         b = b + 1
        #         with open(log_path + '/log.txt', 'a') as log:
        #             log.write("========   test in AD  ==============")
        #             log.write("test_Epoch: {}, iteration: {}\n".format(epoch, b))
        #             log.write("test_SSIM: {}\n".format(ssim_test_value))


if __name__ == '__main__':
    import argparse
    import os
    tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第1块GPU（从0开始）
    # # 定义TensorFlow配置
    # config = tf.ConfigProto()
    # # 配置GPU内存分配方式，按需增长，很关键
    # config.gpu_options.allow_growth = True
    # # 配置可使用的显存比例
    # config.gpu_options.per_process_gpu_memory_fraction = 0.1
    #
    # # 在创建session的时候把config作为参数传进去
    # sess = tf.InteractiveSession(config=config)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='None', help='model to retrain on')
    parser.add_argument('--z_dim', type=str, default='None', help='model comment')
    args = parser.parse_args()#args.z_dim,args.model_name
    train(z_dim=128, model_name=None)

