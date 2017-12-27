import sklearn,numpy as np,pandas as pd,pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets,preprocessing,metrics
from sklearn.model_selection import train_test_split,cross_val_score,validation_curve,learning_curve
import numpy as np,pandas as pd,matplotlib.pyplot as plt,PIL
import matplotlib.pyplot as plt,os
from sklearn.cluster import DBSCAN,KMeans
from sklearn.decomposition import PCA,NMF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import Imputer   #导入预处理模块
from sklearn.metrics import classification_report         #导入预测结果评估模块


n=pd.read_csv('1.txt',header=None)
minx,maxx=n[0].min(),n[0].max()         #确定横坐标的最小值和最大值
X=np.arange(minx,maxx).reshape(-1,1)    #将所有x坐标保存成（-1,1）的数组
plt.scatter(n[0],n[1])                  #绘制面积/房价散点图

linear=sklearn.linear_model.LinearRegression(normalize=False)   
linear.fit(n[0].values.reshape(-1,1),n[1])  #reshape成（-1,1），否则报错
print(linear.coef_,linear.intercept_)       #打印系数和截距
plt.plot(X,linear.predict(X))
plt.grid(True)                          #显示网格

plt.show()
