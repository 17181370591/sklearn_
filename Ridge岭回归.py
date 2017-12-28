import sklearn,numpy as np,pandas as pd,pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets,preprocessing,metrics,linear_model
from sklearn.model_selection import train_test_split,cross_val_score,validation_curve,learning_curve
import numpy as np,pandas as pd,matplotlib.pyplot as plt,PIL
import matplotlib.pyplot as plt,os
from sklearn.cluster import DBSCAN,KMeans
from sklearn.decomposition import PCA,NMF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import Imputer   #导入预处理模块
from sklearn.metrics import classification_report         #导入预测结果评估模块
from sklearn.linear_model import Ridge          #调用岭回归模型

data=pd.read_csv('1.txt',header=None)
x=data.ix[:,1:4]            #读取参数，一共4个
y=data.ix[:,5]              #读取车流量
poly=sklearn.preprocessing.PolynomialFeatures(4)    #6次(多次实验后决定用6次）
x=poly.fit_transform(x)         #将4维参数转为4元6次多项式
train_set_x,test_set_x,train_set_y,test_set_y=\
train_test_split(x,y,test_size=0.3,random_state=0)
        #划分数据为训练集和测试集

clf=Ridge(alpha=1,fit_intercept=True)   #创建岭回归实例
clf.fit(train_set_x,train_set_y)        #训练
sc=clf.score(test_set_x,test_set_y)        #利用测试集计算拟合度
print('sc=',sc)

start,end=200,300                   #绘制拟合曲线 
y_pre=clf.predict(x)            #调用拟合值
time=np.arange(start,end)       #确定横坐标的范围
plt.plot(time,y[start:end],'b',label='real')    #绘制数据点，label是此图像的名字/标签
plt.plot(time,y_pre[start:end],'r',label='predict')     #绘制拟合图像
plt.legend(loc='upper left')            #设置图例(每组图像的名字/标签)的位置
plt.show()
