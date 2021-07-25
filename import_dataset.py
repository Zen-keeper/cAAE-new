import random
import shutil

import numpy as np
import nibabel as nib
import os
import glob,xlwt


from funcs import preproc

def processNii(data):
    X_train_input = []
    for i in data:
        pathx = i
        img = nib.load(pathx).get_data().astype(np.float32)
        img[img < 0] = 0
        img = np.transpose(img, [2, 1, 0])
        # set:无序不重复元素集，去除了完全相同的元素层，就是全是0的
        # idx = [s for s in range(img.shape[0]) if len(set(img[s].flatten())) > 1]
        # img = img[idx]
        _, x, y = img.shape
        max_xy = max(x, y)

        a = int((max_xy - x) / 2)  # 9
        b = int((max_xy - y) / 2)  # 0

        if len(X_train_input) == 0:
            print(img.shape)

        # 表示在二维数组array第一维（此处便是行）前面填充a行，最后面填充a行；
        # 在二维数组array第二维（此处便是列）前面填充b列，最后面填充b列
        img = np.pad(img, ((0, 0), (a, a), (b, b)), mode='edge')
        img = preproc.myresize(img, 128)

        # 归一化-1-1
        img_fdata_flat = img.flatten()
        img_fdata_flat = (img_fdata_flat - np.mean(img_fdata_flat)) / (max(img_fdata_flat) - min(img_fdata_flat))
        img = np.reshape(img_fdata_flat, img.shape)
        img2 = np.flip(img, 1)
        X_train_input.extend(img)
        X_train_input.extend(img2)


        del img_fdata_flat
    return np.asarray(X_train_input)

def create_datasets(retrain=False, task=None, labels=False, ds_scale=0,wd = "./Data/CamCAN_unbiased/CamCAN/T2w",num=None):

    if num is None:
        totalnum = len(os.listdir(wd))
    else:
        totalnum = min(len(os.listdir(wd)),num)
    subject_id = np.array([i for i in glob.glob(wd+"/*") if 'mask' in i and ('.nii' in i or '.img' in i)])
    # subject_train_idx = random.sample(range(0, totalnum), int(totalnum*4/5))
    subject_train_idx = [range(0, len(subject_id))]
    subject_train = subject_id[subject_train_idx]
    print("train :"+str(len(subject_train)))
    # subject_test = [i for i in subject_id if i not in subject_train][:]

    print("retrain is {}".format(retrain))
    print(str(task)+"training subject ids: {}".format(subject_train))
    # print(str(task)+"testing subject ids: {}".format(subject_test))
    return subject_train

    # X_train_input = []
    # X_train_target = []
    # X_train_target_all = []
    # X_dev_input = []
    # X_dev_target = []
    # X_dev_target_all = []
    #
    # for i in subject_train:
    #     print(i)
    #     pathx=i
    #     img = nib.load(pathx).get_data()
    #     img[img<0]=0
    #     img = np.transpose(img, [2, 1, 0])
    #     #set:无序不重复元素集，去除了完全相同的元素层，就是全是0的
    #     # idx = [s for s in range(img.shape[0]) if len(set(img[s].flatten())) > 1]
    #     # img = img[idx]
    #     _, x, y = img.shape
    #     max_xy = max(x,y)
    #
    #     a = int((max_xy-x)/2 )   #   9
    #     b = int((max_xy-y)/2 )   #   0
    #
    #     if len(X_train_input)==0:
    #         print(img.shape)
    #
    #     #表示在二维数组array第一维（此处便是行）前面填充a行，最后面填充a行；
    #     # 在二维数组array第二维（此处便是列）前面填充b列，最后面填充b列
    #     img = np.pad(img, ((0, 0),(a, a), (b,b)), mode='edge')
    #     img = preproc.myresize(img, 128)
    #     #归一化
    #     img_fdata_flat = img.flatten()
    #     img_fdata_flat = (img_fdata_flat - np.mean(img_fdata_flat)) / (max(img_fdata_flat) - min(img_fdata_flat))
    #     img = np.reshape(img_fdata_flat, img.shape)
    #     del img_fdata_flat
    #
    #
    #     if labels:
    #         z = np.genfromtxt(os.path.join(wd, str(i) + '/' + str(i) + "_label.txt"))
    #         assert len(z)==len(img)
    #         X_train_target.extend(z)
    #
    #     if ds_scale!=0:
    #         img = preproc.downsample_image(img[np.newaxis,:,:,:],ds_scale)
    #     X_train_input.extend(img)
    #
    #
    # # for j in subject_test:
    # #     print(j)
    # #     pathx=j
    # #     img = nib.load(pathx).get_data().astype(np.float32)
    # #     img[img < 0] = 0
    # #     img = np.transpose(img, [2, 1, 0])
    # #     # idx = [s for s in range(img.shape[0]) if len(set(img[s].flatten())) > 1]
    # #     # img = img[idx]
    # #     img = np.pad(img, ((0, 0), (a,a), (b,b)), mode='edge')
    # #     img = preproc.myresize(img, 128)
    # #
    # #     if ds_scale!=0:
    # #         img = preproc.downsample_image(img[np.newaxis,:,:,:],ds_scale)
    # #     X_dev_input.extend(img)
    # #     if labels:
    # #         z = np.genfromtxt(os.path.join(wd, str(j) + '/' + str(j) + "_label.txt"))
    # #         print(len(z), img.shape)
    # #         X_dev_target.extend(z)
    #
    # if not labels:
    #     X_train_input = np.asarray(X_train_input)
    #     X_dev_input = np.asarray(X_dev_input)
    #     if(np.isnan(X_train_input).any()):
    #         print("X_train_input数组中有nan值！！！！！！！！")
    #     if (np.isnan(X_dev_input).any()):
    #         print("X_dev_input  数组中有nan值！！！！！！！！")
    #     return np.nan_to_num(X_train_input)#, np.nan_to_num(X_dev_input)
    # else:
    #     return X_train_input, X_train_target, X_dev_input, X_dev_target

def create_test(wd = "./Data/CamCAN_unbiased/CamCAN/T2w/",ds_scale = 0):
    if(isinstance(wd,str)):
        paths = os.listdir(wd)
    else :
        paths = wd
    for item in paths:
        # p = os.path.join(wd,item)
        # p = glob.glob(temp+"/*/mask*.img")
        p = item
        # if item.endswith(".hdr"):
        #     continue
        x_test = []
        if (isinstance(wd, str)):
            p = os.path.join(wd,item)
        # else:
        #     p = item
        img = nib.load(p).get_data().astype("float32")
        img[img < 0] = 0
        img = np.transpose(img, [2, 1, 0])
        # set:无序不重复元素集，去除了完全相同的元素层，就是全是0的
        # idx = [s for s in range(img.shape[0]) if len(set(img[s].flatten())) > 1]
        # img = img[idx]
        _, x, y = img.shape
        max_xy = max(x, y)

        a = int((max_xy - x) / 2)  # 9
        b = int((max_xy - y) / 2)  # 0
        # idx = [s for s in range(img.shape[0]) if len(set(img[s].flatten())) > 1]
        # img = img[idx]
        img = np.pad(img, ((0, 0), (a, a), (b, b)), mode='edge')
        img = preproc.myresize(img, 128)
        # 归一化
        img_fdata_flat = img.flatten()
        img_fdata_flat = (img_fdata_flat - np.mean(img_fdata_flat)) / (max(img_fdata_flat) - min(img_fdata_flat))
        img = np.reshape(img_fdata_flat, img.shape)

        if ds_scale != 0:
            img = preproc.downsample_image(img[np.newaxis, :, :, :], ds_scale)
        x_test.extend(img)
        x_test = np.array(x_test)
        if (np.isnan(x_test).any()):
            print("读取测试数据"+wd+":x_test数组中有nan值！！！！！！！！")
            # shutil.copy(p,"C:\\Users\jsc\Desktop\\bad")
            # shutil.move(p,"C:\\Users\jsc\Desktop\\bad")
            continue
        yield item.split("mask")[-1].split('.')[0], np.nan_to_num(x_test)

#用于新的文件目录形如：
#E:\AD\AD_file\002_S_0816_2008-01-28\mri\*.nii
def create_test_AD(wd = "./Data/CamCAN_unbiased/CamCAN/T2w/",ds_scale = 0):
    if(isinstance(wd,str)):
        paths = os.listdir(wd)

    else :
        paths = wd
    date = "1"
    for item in paths:
        if item.endswith(".hdr"):
            continue
        x_test = []
        if (isinstance(wd, str)):
            # 这两行是视情况加
            # date = os.listdir(os.path.join(wd, item))[0]
            if(date=="residual"):
                continue
            # name = os.path.join(wd, item)
            p = os.path.join(wd, item)
            # name = glob.glob(p+"\\mask*.img")
            # name.extend(glob.glob(p + "\\mask*.nii"))
            # if(len(name)<1):
            #     continue
            # p = name[0]
        else:
            p = item
        nii_img = nib.load(p)
        img = nii_img.get_data().astype(np.float32)
        affine = nii_img.affine.copy()
        hdr = nii_img.header.copy()
        img[img < 0] = 0
        img = np.transpose(img, [2, 1, 0])
        # set:无序不重复元素集，去除了完全相同的元素层，就是全是0的
        # idx = [s for s in range(img.shape[0]) if len(set(img[s].flatten())) > 1]
        # img = img[idx]
        _, x, y = img.shape
        max_xy = max(x, y)

        a = int((max_xy - x) / 2)  # 9
        b = int((max_xy - y) / 2)  # 0
        # idx = [s for s in range(img.shape[0]) if len(set(img[s].flatten())) > 1]
        # img = img[idx]
        img = np.pad(img, ((0, 0), (a, a), (b, b)), mode='edge')
        img = preproc.myresize(img, 128)
        # 归一化
        img_fdata_flat = img.flatten()
        img_fdata_flat = (img_fdata_flat - np.mean(img_fdata_flat)) / (max(img_fdata_flat) - min(img_fdata_flat))
        img = np.reshape(img_fdata_flat, img.shape)

        if ds_scale != 0:
            img = preproc.downsample_image(img[np.newaxis, :, :, :], ds_scale)
        x_test.extend(img)
        x_test = np.array(x_test)
        if (np.isnan(x_test).any()):
            print("读取测试数据"+wd+":x_test数组中有nan值！！！！！！！！")
            # shutil.copy(p,"C:\\Users\jsc\Desktop\\bad")
            shutil.move(p,"C:\\Users\jsc\Desktop\\bad")

            continue
        yield item,x_test,affine,hdr # +"-"+date

if __name__== '__main__':
    z_dim = 128
    # create_datasets(retrain=False, task="aae_wgan_" + str(z_dim),ds_scale=0)
    # create_test()
    create_datasets()