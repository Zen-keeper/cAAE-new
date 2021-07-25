import os

import tensorflow as tf
import tensorlayer as tl
from tensorflow.python.framework import graph_util
import argparse
from model_conv4 import encoder, decoder, discriminator
import import_dataset as datasets
from funcs.preproc import *
import numpy as np
import pandas as pd
from PIL import Image
import nibabel as nib

parser = argparse.ArgumentParser()
# parser.add_argument('--image', default='', type=str,
#                     help='The filename of image to be completed.')
# parser.add_argument('--mask', default='', type=str,
#                     help='The filename of mask, value 255 indicates mask.')
# parser.add_argument('--output', default='output.png', type=str,
#                     help='Where to write output.')#Adversarial_Autoencoder/cAdversarial_Autoencoder_WGAN/Saved_models
parser.add_argument('--checkpoint_dir', default='D:\python_file\cAAE-master\cAAE-master\Results\conv4_selecttrain\z_0\\model\\', type=str,
                    help='The directory of tensorflow checkpoint.')
parser.add_argument('--in_server', default=False, type=bool,
                    help='Is the test in server or in local windows environment.')
parser.add_argument('--data_path', default="D:\work\AD_V3\image_class\ori\\", type=str,
                    help='The class of the test.')
parser.add_argument('--data_class', default="AD", type=str,
                    help='The class of the test.')
# Parameters
BATCH_SIZE = 91
EPOCHS = 200
results_path = './Results/Adversarial_Autoencoder'

model_path = "./Results/c5model"

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


def my_te(z_dim=128, model_name=None):
    # f = xlwt.Workbook()
    # sheet_ad = f.add_sheet("scd_average_residual", cell_overwrite_ok=True)
    # sheet_nc = f.add_sheet("nc_average_residual", cell_overwrite_ok=True)
    args = parser.parse_args()
    sess = tf.Session()
    x_input = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 128, 128, 1], name='Input')
    df = pd.read_excel("C:\\Users\jsc\Desktop\\xlss\\ori_gen_version412a.xls", sheet_name="v6ad")
    data = df.values[:,0]
    names = []
    for n in data:
        names.append(str.split(n,"-")[0][0:10])
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
    ssim_res = []


    for (name, batch,affine, hdr) in datasets.create_test_AD(args.data_path+args.data_class):
        # if(not name[0:10] in names):
        #     continue
        list_slice = []

        ssims = []
        # sheet_ad.write(object_index, 0, name)
        batch_x = batch[0:batch_size,:,:]
        batch_x = batch_x[:, :, :, np.newaxis]

        de_out = sess.run(out, feed_dict={x_input: batch_x})
        gen = np.array(de_out)

        for c in range(0,len(gen)):
            # if not os.path.exists("./Results/residual"):
            #     os.mkdir("./Results/residual")
            # if not os.path.exists("./Results/residual/"+args.data_class):
            #     os.mkdir("./Results/residual/"+args.data_class)
            # save_img(batch_x[c, :, :, 0],gen[c, :, :, 0],path="./Results/residual/scd/"+ name + str(c) + ".bmp")

            # ssims.append(skimage.measure.compare_ssim(batch_x[c, :, :, 0], gen[c, :, :, 0], data_range=1))
            b = cv2.resize(batch_x[c, :, :, 0], (109, 109), interpolation=cv2.INTER_AREA)
            b = crop_center(b, 91, 109)

            g = cv2.resize(gen[c, :, :, 0], (109, 109), interpolation=cv2.INTER_AREA)
            g = crop_center(g, 91, 109)
            # g = np.flip(g,1)

            # res = np.subtract(g,b)
            # res[res<0] = 0
            list_slice.append(b)#是否要加g绝对值？？

        list_slice = np.array(list_slice)
        list_slice = np.swapaxes(list_slice,0,2)
        # nii_img = nib.load(args.data_path + args.data_class + '/'+ name + ".nii")

        new_image = nib.Nifti1Image(list_slice, affine, hdr)
        # if not os.path.exists(args.data_path+args.data_class+"res/"+name+"/residual"):
        #     os.makedirs(args.data_path+args.data_class+"res/"+name+"/residual")
        if not os.path.exists(args.data_path+args.data_class+"ori/"):
            os.makedirs(args.data_path+args.data_class+"ori/")
        # nib.save(new_image,args.data_path+'/'+args.data_class+"/"+name+"/residual/"+name+ '_res.nii')
        # nib.save(new_image, "E:\AD\\ad\\residual\\" + name+"_"+date + '_res.nii')
        # nib.save(new_image, "E:\AD\\ad\\ori\\" + name+"_"+date + '_ori.nii')
        nib.save(new_image, args.data_path+args.data_class+"ori/"+ name + '_ori.nii')
        # nib.save(new_image, "F:\MCI\mcifor_c\\trans\\ori\\" + name + '_ori.nii')
        # temp_ssim = np.mean(np.mean(ssims))
        # sheet_nc.write(object_index, 1, name)
        # sheet_nc.write(object_index, 2, temp_ssim)
        # print(temp_ssim)
        # ssim_res.append(np.mean(ssims))
        # object_index = object_index + 1
        # f.save('nc_all_128-NEW.xls')

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
