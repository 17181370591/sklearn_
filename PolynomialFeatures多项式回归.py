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

n=pd.read_csv('1.txt',header=None)
dx=n[0].values.reshape(-1,1)
dy=n[1]
X=np.arange(dx.min(),dx.max()).reshape(-1,1)

'''
sklearn的多项式回归，实际上是先将变量X处理成多项式特征，然后使用线性模型
学习多项式特征的参数
'''

poly_reg=sklearn.preprocessing.PolynomialFeatures(degree=2)             #degree表示生成2次多项式
X_poly=poly_reg.fit_transform(dx)           #用dx生成多项式？
lin_reg_2=linear_model.LinearRegression()
lin_reg_2.fit(X_poly,dy)                #通过训练生成多项式的参数？

plt.scatter(dx,dy)
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)))
            #先生成X的多项式，再用lin_reg_2预测(通过生成的参数计算)y值
plt.show()


