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
from sklearn.decomposition import PCA

x,y=datasets.load_iris(return_X_y=True) #返回元祖
pca=PCA(n_components=2)             #降至2维
re_x=pca.fit_transform(x)           #对鸢尾花的data进行训练
dic={}
dicc={}
for i in range(y.size):
    if y[i] in dic.keys(): 
        dic[y[i]].append(re_x[i][0])
        dicc[y[i]].append(re_x[i][1])
    else:
        dic[y[i]]=[]
        dic[y[i]].append(re_x[i][0])
        dicc[y[i]]=[]
        dicc[y[i]].append(re_x[i][1])
for i in set(y):
    plt.scatter(dic[i],dicc[i])
plt.show()
