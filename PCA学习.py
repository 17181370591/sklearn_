import sklearn,numpy as np,pandas as pd,pickle
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets,preprocessing,metrics
from sklearn.model_selection import train_test_split,cross_val_score,validation_curve,learning_curve
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA,NMF

n_row,n_col=3,3
n_components=n_row*n_col
image_shape=(64,64)
data1=datasets.fetch_olivetti_faces(shuffle=True,random_state=np.random.RandomState(0))
faces=data1.data
def plot_gallery(title,images,n_col=n_col,n_row=n_row):
    plt.figure(figsize=(2*n_col,2.26*n_row))    #创建图片，指定图片大小
    plt.suptitle(title,size=16)       #设置标题和字号大小
    for i,comp in enumerate(images):
        plt.subplot(n_row,n_col,i+1)        #绘制子图
        vmax=max(comp.max(),-comp.min())
        plt.imshow(comp.reshape(image_shape),cmap=plt.cm.gray,
                   interpolation='nearest',vmin=-vmax,vmax=vmax)
        #绘图，cmap还是的参数都可以省去，cmap=plt.cm.gray表示使用灰度图
        plt.xticks(())
        plt.yticks(())                          #取出子图的坐标轴标签
    plt.subplots_adjust(0.01,0.05,0.99,0.93,0.04,0)     #调整位置 
estimators=[
    ('Eigenfaces-PCA using randomized SVD',
     sklearn.decomposition.PCA(n_components=7,whiten=True)),
    ('Non-negative components-NMF',
    sklearn.decomposition.NMF(n_components=5,init='nndsvda',tol=5e-3))]
    #文本部分是name，分别创建PCA和NMF对象,n_components是取多少个数据
for name,estimator in estimators:
    estimator.fit(faces)          #通过遍历训练
    components_=estimator.components_
    plot_gallery(name,components_[:n_components])
plt.show()    
