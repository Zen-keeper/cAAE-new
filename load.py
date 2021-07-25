import os

import tensorflow as tf

import tensorlayer as tl
from tensorflow.python.framework import graph_util
import argparse
from model_conv4 import encoder, decoder, discriminator
import import_dataset as datasets
from funcs.preproc import *
from scipy import io
import matplotlib.pylab as plt
import numpy as np
from PIL import Image
import nibabel as nib

parser = argparse.ArgumentParser()
# parser.add_argument('--image', default='', type=str,
#                     help='The filename of image to be completed.')
# parser.add_argument('--mask', default='', type=str,
#                     help='The filename of mask, value 255 indicates mask.')
# parser.add_argument('--output', default='output.png', type=str,
#                     help='Where to write output.')#Adversarial_Autoencoder/cAdversarial_Autoencoder_WGAN/Saved_models
# parser.add_argument('--checkpoint_dir', default='./Results/c5/model/', type=str,
#                     help='The directory of tensorflow checkpoint.')
parser.add_argument('--checkpoint_dir', default='D:\python_file\cAAE-master\cAAE-master\Results\conv4_selecttrain\z_0\\model\\', type=str,
                    help='The directory of tensorflow checkpoint.')
# parser.add_argument('--checkpoint_dir', default='D:\python_file\cAAE-master\cAAE-master\Results\\tmp\model\\', type=str,
#                     help='The directory of tensorflow checkpoint.')
parser.add_argument('--in_server', default=False, type=bool,
                    help='Is the test in server or in local windows environment.')
parser.add_argument('--data_path', default="D:\work\\xuanwu_data\\", type=str,
                    help='The class of the test.')
parser.add_argument('--data_class', default="HC_xuanwu", type=str,
                    help='The class of the test.')
# Parameters
BATCH_SIZE = 91
EPOCHS = 200
results_path = './Results/Adversarial_Autoencoder'

model_path = "./Results/c5model"
# class testModel(Model):
#     def __init__(self):
#         super().__init__('InpaintModel')
#
#     def build_inpaint_net(self, x, mask, config=None, reuse=False,
#                           training=True, padding='SAME', name='inpaint_net'):

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]
def connect_model(x_input, z_dim=128,reuse=False, is_train=True):
    encoder_out = encoder(x_input,z_dim,reuse = reuse,is_train=is_train)
    decoder_out,std_ = decoder(encoder_out,reuse = reuse,is_train=is_train)
    return decoder_out
def save_img(ori,gen,path):
    b = cv2.resize(ori, (109, 109), interpolation=cv2.INTER_AREA)
    b = crop_center(b, 91, 109)

    g = cv2.resize(gen, (109, 109), interpolation=cv2.INTER_AREA)
    g = crop_center(g, 91, 109)

    outputImg = Image.fromarray(np.abs(np.subtract(b, g) * 255.0))
    outputImg = outputImg.convert('L')
    outputImg.save(path)

import xlwt
def my_te(z_dim=128, model_name=None):
    f = xlwt.Workbook()

    sheet_nc = f.add_sheet("nc_average_residual", cell_overwrite_ok=True)
    args = parser.parse_args()
    sess = tf.Session()
    x_input = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 128, 128, 1], name='Input')
    # encoder_out = encoder(x_input,z_dim, is_train=True)
    # decoder_out, std_ = decoder(encoder_out, is_train=True)
    out = connect_model(x_input,z_dim)
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    if args.in_server:
        checkpoint_dir = "/data/test/CAAE3/Results/Adversarial_Autoencoder/cAdversarial_Autoencoder_WGAN/Saved_models/"
    else :
        checkpoint_dir = args.checkpoint_dir
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(
            checkpoint_dir, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded.')

    batch_size = BATCH_SIZE
    object_index = 1
    # sheet_ad.write(0, 1, "scd残差平均值")
    # for (name, batch) in datasets.create_test("D:/work/AD_V3/"+args.data_class):
    ssim_res = []
    for (name, batch) in datasets.create_test(args.data_path+args.data_class):
        # for i in range(len(batch)):
        list_slice = []
        ssims = []
        sheet_nc.write(object_index, 0, name)
        batch_x = batch[0:batch_size,:,:]
        batch_x = batch_x[:, :, :, np.newaxis]

        de_out = sess.run(out, feed_dict={x_input: batch_x})
        gen = np.array(de_out)
        # batch_x = batch_x.astype(np.float32)
        # residual = batch_x-gen
        # ave = np.average(residual)
        # sheet_ad.write(object_index,1,float(ave))
        # sheet_ad.write(object_index, 6, float(np.average(batch_x)))
        for c in range(0,len(gen)):
            # if not os.path.exists("./Results/residual"):
            #     os.mkdir("./Results/residual")
            # if not os.path.exists("./Results/residual/"+args.data_class):
            #     os.mkdir("./Results/residual/"+args.data_class)
            # save_img(batch_x[c, :, :, 0],gen[c, :, :, 0],path="./Results/residual/scd/"+ name + str(c) + ".bmp")
            ssims.append(skimage.measure.compare_ssim(batch_x[c, :, :, 0], gen[c, :, :, 0],data_range=1))
            b = cv2.resize(batch_x[c, :, :, 0], (109, 109), interpolation=cv2.INTER_AREA)
            b = crop_center(b, 91, 109)
            # plt.imshow(batch_x[c, :, :, 0])
            # plt.waitforbuttonpress()
            # plt.imshow(gen[c, :, :, 0])
            # plt.waitforbuttonpress()
            g = cv2.resize(gen[c, :, :, 0], (109, 109), interpolation=cv2.INTER_AREA)
            g = crop_center(g, 91, 109)
            # g = np.flip(g,1)

            res = np.subtract(g,b)
            res[res<0] = 0
            list_slice.append(g)#是否要加绝对值？？

        list_slice = np.array(list_slice)
        list_slice = np.swapaxes(list_slice, 0, 2)
            # nii_img = nib.load(args.data_path + args.data_class + '/'+ name + ".nii")
        nib.save(list_slice,args.data_path + args.data_class + 'res/'+ name + "res.nii")



        temp_ssim = np.mean(np.mean(ssims))
        sheet_nc.write(object_index, 2, temp_ssim)
        print(temp_ssim)
        ssim_res.append(np.mean(ssims))
        object_index = object_index + 1
        # f.save('xuanwu_newtraindata.xls')


            # if(object_index==1):
            # else:
            #     list_slice[object_index] = list_slice[object_index]*object_index/(object_index+1)+ np.subtract(batch_x[c, :, :, 0],gen[c, :, :, 0])/(object_index+1)
        # if(not os.path.exists("./Results/residual/three3D/{}".format(args.data_class))):
        #     os.mkdir("./Results/residual/three3D/{}".format(args.data_class))
        # np.save("./Results/residual/three3D/{}/test{}_{}.npy".format(args.data_class,str(object_index),name), list_slice)


    print("The result of mean ssim is,",str(np.mean(ssim_res)))
        # list_slice = np.array(list_slice)
        # list_slice = np.swapaxes(list_slice,0,2)
        # nii_img = nib.load(args.data_path+args.data_class+name+".nii")
        # affine = nii_img.affine.copy()
        # hdr = nii_img.header.copy()
        # new_image = nib.Nifti1Image(list_slice, affine, hdr)
        # # new_image = nib.Nifti1Image(list_slice, np.eye(4))
        # if not os.path.exists(args.data_path+"residual/"+args.data_class):
        #     os.makedirs(args.data_path+"residual/"+args.data_class)
        # nib.save(new_image,args.data_path+"residual/"+args.data_class+"/"+name+ '.nii.gz')
        # # sheet_ad.write(object_index, 4, "宣武正常人平均残差")
        # # object_nums = 0
        # # for (name, batch) in datasets.create_test("./Data/selscthc"):
        # #     # for i in range(len(batch)):
        # #     object_nums =object_nums+1
        # #     if object_nums > 200:
        # #         break
        # #     sheet_ad.write(object_index, 3, name)
        # #     batch_x = batch[0:batch_size,:,:]
        # #     batch_x = batch_x[:, :, :, np.newaxis]
        # #
        # #     de_out = sess.run(out, feed_dict={x_input: batch_x})
        # #     gen = np.array(de_out)
        # #     batch_x = batch_x.astype(np.float32)
        # #     residual = batch_x - gen
        # #     ave = np.average(residual)
        # #     sheet_ad.write(object_index,4,float(ave))
        # #     sheet_ad.write(object_index,8,float(np.average(batch_x)))
        # #     for c in range(gen.shape[0]):
        # #         # plt.subplot(121)
        # #         # plt.imshow(batch_x[c, :, :, 0])
        # #         # plt.subplot(122)
        # #         # plt.imshow(gen[c, :, :, 0])
        # #         # # ssim = skimage.measure.compare_ssim(batch_x[c, :, :, 0], gen[c, :, :, 0], data_range=1.0)
        # #         # plt.waitforbuttonpress()
        # #         if not os.path.exists("./Results/residual"):
        # #             os.mkdir("./Results/residual")
        # #         if not os.path.exists("./Results/residual/NC"):
        # #             os.mkdir("./Results/residual/NC")
        # #         save_img(batch_x[c, :, :, 0],gen[c, :, :, 0],path="./Results/residual/NC/"+ name + str(c) + ".bmp")
        # object_index = object_index + 1

        # # 写入表格
        # for ceng in range(len(list_slice)):
        #     for row in range(128):
        #         for col in range(128):
        #             sheet_ad.write(row+1+128*ceng,col+1,list_slice[ceng][row,col])
        # f.save("./scd_residual.xls")


if  __name__== '__main__':
    # freeze_graph("D:/python_file/cAAE-master/cAAE-master/Results/c1model/")
    # graph = load_graph("D:/python_file/cAAE-master/cAAE-master/Results/c1model/frozen_model.pb")
    my_te(128)
