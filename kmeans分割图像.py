import sklearn,numpy as np,pandas as pd,pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets,preprocessing,metrics
from sklearn.model_selection import train_test_split,cross_val_score,validation_curve,learning_curve
import numpy as np,pandas as pd,matplotlib.pyplot as plt,PIL
import matplotlib.pyplot as plt,os
from sklearn.cluster import DBSCAN,KMeans
from sklearn.decomposition import PCA,NMF
data=[]
def pic(f):     #打开图像的方法，此处没用
    plt.imshow(PIL.Image.open(open(f,'rb')))
    plt.show()
def fkmeans(cl):        #kmean分割
    f='1.jpg'
    f=open(f,'rb')          #读取图像成2进制数据
    img=PIL.Image.open(f)       #pil打开图像数据
    m,n=img.size             #获取图片大小
    data=np.mat([(img.getpixel((i,j))) for i in range(m) for j in range(n)])/256
            #读取每个像素点的（r，g，b）到矩阵，除以256使所有值范围为[0,1)
    km=KMeans(n_clusters=cl)
    label=km.fit_predict(data).reshape([m,n])
            #训练数据成cl个，reshape成图像大小
    pic_new=PIL.Image.new('L',(m,n))
            #新建空白图像，大小为（m，n），选择亮度模式？
    for i in range(m):
        for j in range(n):
            pic_new.putpixel((i,j),int(256/(label[i][j]+1)))
                    #为每个像素分配值（必须是整数），int里作用是使cl个区域两两间距离是256/cl
    pic_new.save('2.jpg','JPEG')   

fkmeans(5)
##f='3.jpg'
##f=open(f,'rb')      
##img=PIL.Image.open(f)       
##m,n=img.size           
##data=np.mat([(img.getpixel((i,j))) for i in range(m) for j in range(n)])/256
##db=DBSCAN(eps=0.02,min_samples=100)
##print(1)
##label1=db.fit_predict(data)
##print(2)
##label=label1.reshape([m,n])
##print(3)
##pic_new=PIL.Image.new('L',(m,n))
##for i in range(m):
##    
##    for j in range(n):
##        #print(i,j,label[i][j]+1)
##        pp=int(256/(label[i][j]+2))
##        #print(pp)
###        pic_new.putpixel((i,j),int(256/(label[i][j]+1)))         
##        pic_new.putpixel((i,j),pp)
##pic_new.save('2.jpg','JPEG')
##上面尝试用dbscan分割图像，效果不好        

