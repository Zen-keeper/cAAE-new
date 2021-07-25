import numpy as np
import skimage.measure
import scipy
# from scipy.misc import imresize
import matplotlib.pylab as plt
from skimage.transform import resize
import cv2

def generate_patch(img_3d, patch_size, step_size, threshold):
    img = np.asarray(img_3d)
    min_ = img.min()
    l_patch = []
    _,x,y = img.shape
    for im in img:
        for xidx in np.arange(0, x-patch_size+step_size, step_size):
            for yidx in np.arange(0, y-patch_size+step_size, step_size):
               patch = im[xidx:xidx+patch_size, yidx:yidx+patch_size]
               if (patch==min_).sum()<patch_size**2*threshold:
                   l_patch.append(patch)
    return np.asarray(l_patch)

## AvgPool-like effect
def generate_low_resolution_images(img_3d):
    _,original_x, orignal_y = img_3d.shape
    img_3d_lowres = []
    for im in img_3d:
        im_lowres = skimage.measure.block_reduce(im, (2,2), np.mean)#通过将函数应用于本地块来下采样图像。
        img_3d_lowres.append(im_lowres)
    return np.asarray(img_3d_lowres)

def myresize(img_3d,scale):
    img = img_3d
    res = []
    for im in img:
        # i_ = resize(im, (scale,scale))#0-1：float64
        i_re = cv2.resize(im,(128,128))
        # i_re = i_*(im.max()-im.min())+im.min()
        # i_re = np.zeros_like(i_)-1
        # if(i_.max()-i_.min()!=0):
        #     i_re = (i_-i_.min())*2.0/(i_.max()-i_.min())-1
            # plt.imshow(i_re)
        # plt.imshow(np.concatenate(np.asarray(i_),np.asarray(i_re)))
        # plt.waitforbuttonpress()
        res.append(i_re)
    return np.asarray(res)

def downsample_image(dt, ds_scale=3):
    res = []
    for img in dt:
        init_scale=0
        while init_scale<ds_scale:
            img = generate_low_resolution_images(img)#下采样
            init_scale+=1
        res.extend(img)
    return np.asarray(res)

