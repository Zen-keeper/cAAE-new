# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 13:58:37 2020

TENE

@author: jsc
"""

#import numpy as np
#from sklearn.manifold import TSNE
#X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
#tsne = TSNE(n_components=2)
#tsne.fit_transform(X)
#print(tsne.embedding_)


# coding='utf-8'
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE
import os
import nibabel as nib
from PIL import Image


def get_data():
    ad = []
#    nc = []
#    scd_ori = []
#    scd_gen = []
    label = []
    msker = []
    # path_AD = "D:\python_file\cAAE-master\cAAE-master\Results\c3\img/nc\gen/"
    path_AD = "D:/work/AD_V3/slice/"
    path_NC = "D:/work/ADNI_AV45_TI_HC_MCI_AD/ADNI_CN/slice/"
    # path_SCD_ori = "D:\python_file\cAAE-master\cAAE-master\Results\c3\img/ad\ori/"
    # path_SCD_gen = "D:\python_file\cAAE-master\cAAE-master\Results\c3\img/ad\gen/"
    name_ADs = os.listdir(path_AD)
    name_NCs = os.listdir(path_NC)
    # name_SCDs = os.listdir(path_SCD_ori)
    # name_scd_gens = os.listdir(path_SCD_gen)
    for name_nc in name_NCs:
        # img = nib.load(path_AD+name_ad).get_data()
        img = np.array(Image.open(path_NC+name_nc))
        # img[img < 0] = 0
        # x_min, x_max = np.min(img, 0), np.max(img, 0)
        # if (x_max.any() == x_min.any()):
        #     continue
        # img = (img - x_min) / (x_max - x_min)
        ad.append(img.flatten())
        if name_nc.split("_")[0]=="gen":
            continue
            label.append('r')
            msker.append('+')
        else:
            label.append('g')
            msker.append('+')

    for name_ad in name_ADs:
        # img = nib.load(path_AD+name_ad).get_data()
        img = np.array(Image.open(path_AD+name_ad))
        # img[img < 0] = 0
        # x_min, x_max = np.min(img, 0), np.max(img, 0)
        # if (x_max.any() == x_min.any()):
        #     continue
        # img = (img - x_min) / (x_max - x_min)
        ad.append(img.flatten())
        if name_ad.split("_")[0]=="gen":
            continue
            label.append('b')
            msker.append('*')
        else:
            label.append('chocolate')
            msker.append('*')

    # for name_nc in name_NCs:
    #     img = nib.load(path_NC+name_nc).get_data()
    #     # img[img < 0] = 0
    #     # x_min, x_max = np.min(img, 0), np.max(img, 0)
    #     # if(x_max==x_min):
    #     #     continue
    #     # img = (img - x_min) / (x_max - x_min)
    #     ad.append(img.flatten())
    #     label.append('g')

        
    # for name_ad in name_SCDs:
    #     img = nib.load(path_SCD_ori+name_ad).get_data()
    #     # img[img < 0] = 0
    #     # x_min, x_max = np.min(img, 0), np.max(img, 0)
    #     # if (x_max == x_min):
    #     #     continue
    #     # img = (img - x_min) / (x_max - x_min)
    #     ad.append(img.flatten())
    #     label.append('b')
    #
    # for name_nc in name_scd_gens:
    #     img = nib.load(path_SCD_gen+name_nc).get_data()
    #     # img[img < 0] = 0
    #     # x_min, x_max = np.min(img, 0), np.max(img, 0)
    #     # if (x_max == x_min):
    #     #     continue
    #     # img = (img - x_min) / (x_max - x_min)
    #     ad.append(img.flatten())
    #     label.append('chocolate')
    return np.nan_to_num(np.array(ad)) ,label,msker

def plot_embedding(data, label, title):
    fig = plt.figure()
    ax = plt.subplot(111)

    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 # color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 3})
    
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def main():
    all_img,label,maker= get_data()
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2,perplexity=50.0,init='pca', random_state=50,n_iter=1000)
    t0 = time()
    result = tsne.fit_transform(all_img)
    # fig = plot_embedding(result, label,
    #                      't-SNE embedding of the digits (time %.2fs)'
    #                      % (time() - t0))
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 创建子图
    plt.scatter(result[:, 0], result[:, 1],s=5, c=label,marker='+')
    plt.scatter([],[],s=5, c='g',marker='+')
    plt.scatter([],[],s=5, c='b',marker='+')
    plt.scatter([],[],s=5, c='chocolate',marker='+')
    plt.legend(['NC_gen','NC_ori','AD_gen','AD_ori'])
    # 去掉边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    plt.show()


if __name__ == '__main__':
    main()
