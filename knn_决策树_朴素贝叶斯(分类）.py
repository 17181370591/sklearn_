import sklearn,numpy as np,pandas as pd,pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets,preprocessing,metrics
from sklearn.model_selection import train_test_split,cross_val_score,validation_curve,learning_curve
import numpy as np,pandas as pd,matplotlib.pyplot as plt,PIL
import matplotlib.pyplot as plt,os
from sklearn.cluster import DBSCAN,KMeans
from sklearn.decomposition import PCA,NMF
from sklearn.naive_bayes import GaussianNB
def f1():
    x=[[0],[1],[2],[3]]
    y=[0,0,1,1]
    neigh=KNeighborsClassifier(n_neighbors=3)
    #取距离某点最近的3个点，他们大多数属于哪一类这个点就属于哪一类
    neigh.fit(x,y)
    print('neigh.predict(1.5)=',neigh.predict(1.5))     #预测1.5》》0
    print('neigh.predict(1.51)=',neigh.predict(1.51))     #预测1.5》》1
def f2():
    iris=datasets.load_iris()
    clf=sklearn.tree.ExtraTreeClassifier()
    cs=cross_val_score(clf,iris.data,iris.target,cv=10)
    print('cs=',cs)
    #看不懂：决策树
    clf.fit(iris.data,iris.target)
    cs1=clf.predict(iris.data)==iris.target
    print('cs1=',cs1)
def f3():
    x=np.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
    y=np.array([1,1,1,2,2,2])
    clf=GaussianNB(priors=None)
    clf.fit(x,y)
    print(clf.predict([[-0.8,-1]]))
    #看不懂：朴素贝叶斯分类

f3()
