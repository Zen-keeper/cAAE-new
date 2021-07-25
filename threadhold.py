import cv2,os
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_circles
import matplotlib.pyplot as plt
import nibabel as nib
import time
import numpy as np
from sklearn.cluster import KMeans

from sklearn.cluster import DBSCAN

def myDBSCAN(ori,X,img):
    # DBSCAN 算法
    # img = np.transpose(img)
    # ori = np.transpose(ori)
    t0 = time.time()
    dbscan = DBSCAN(eps=4, min_samples=10).fit(X)  # 该算法对应的两个参数
    t = time.time() - t0
    plt.subplot(131)
    plt.imshow(ori,cmap="gray")
    plt.title("residual")
    plt.subplot(132)
    plt.imshow(img,cmap="gray")
    plt.title("residual_withthreshold")
    plt.subplot(133)
    # plt.xlim(0, 91)  # X轴范围
    # plt.ylim(109, 0)  # 显示y轴范围
    plt.imshow(img, cmap="gray")
    plt.scatter(X[:, 0], X[:, 1],s=10, c=dbscan.labels_)
    plt.title("residual_cluster")
    plt.show()
    plt.waitforbuttonpress()
def cv2fringe(ori,img):
    th4 = img
    plt.subplot(131)
    plt.title("ori")
    plt.imshow(ori, cmap="gray")

    ret, binary = cv2.threshold(th4, 30, 255, cv2.THRESH_BINARY)
    plt.subplot(132)
    plt.title("binary")
    plt.imshow(binary, cmap="gray")
    bgr = cv2.cvtColor(th4, cv2.COLOR_GRAY2BGR)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # bgr = np.full_like(bgr,255)
    # cv2.drawContours(bgr, contours, -1, (255, 0, 0),1)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # print(cv2.contourArea(c))
        area_sum = 0
        for i in range(x, x + w + 1):
            for j in range(y, y + h + 1):
                area_sum = area_sum + th4[j][i]
                # print(str(i)+"  "+ str(j) +"  "+str(th4[j][i]))
        print(area_sum)
        if (area_sum > 800):
            cv2.rectangle(bgr, (x, y), (x + w, y + h), (255, 0, 0), 1)

    plt.subplot(133)
    plt.title("bgr")
    plt.imshow(bgr)

    plt.show()
def split_threadhold(path="D:\work\AD_V3\image_class\ori"):
    image_paths = os.listdir(path)
    for image in image_paths:
        dir = os.path.join(path,"residual_002_S_5018.nii")


        # 灰度图读入
        # img = np.flipud(cv2.imread(dir, 0)) # 0:灰度，1：三通道
        img = nib.load(dir).get_data()
        img = img[:, :, 60]*255
        # 阈值分割
        # # 应用5种不同的阈值方法
        # ret, th1 = cv2.threshold(img, 55, 255, cv2.THRESH_BINARY)
        # ret, th2 = cv2.threshold(img, 55, 255, cv2.THRESH_BINARY_INV)
        # ret, th3 = cv2.threshold(img, 55, 255, cv2.THRESH_TRUNC)
        ret, th4 = cv2.threshold(img, 33, 255, cv2.THRESH_TOZERO)
        # ret, th5 = cv2.threshold(img, 55, 255, cv2.THRESH_TOZERO_INV)
        X=[]
        for i in range(len(th4)):
            for j in range(len(th4[0])):
                if(th4[i,j]>0):
                    X.append(np.array([j,i]))
        if len(X)!=0:
            myDBSCAN(img,np.array(X) ,th4)
        # cv2fringe(img,th4)





def adaptive_th(path="D:\work\AD_V3\image_class\ori"):
    image_paths = os.listdir(path)
    for image in image_paths:
        dir = os.path.join(path, "residual_002_S_5018.nii")


        # 自适应阈值对比固定阈值
        img = cv2.imread(dir, 0)  # 0:灰度，1：三通道

        # 固定阈值
        ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        # 自适应阈值
        imgrev = cv2.bitwise_not(img) #颜色反转
        th2 = cv2.adaptiveThreshold(
            imgrev, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 25)
        th2 = cv2.bitwise_not(th2)
        th3 = cv2.adaptiveThreshold(
            imgrev, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 25)
        th3 = cv2.bitwise_not(th3)
        titles = ['Original', 'Global(v = 127)', 'Adaptive Mean', 'Adaptive Gaussian']
        images = [img, th1, th2, th3]

        for i in range(4):
            plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
            plt.title(titles[i], fontsize=8)
            plt.xticks([]), plt.yticks([])
        plt.show()






if __name__=="__main__":
    split_threadhold()


