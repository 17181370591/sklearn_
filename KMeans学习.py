import sklearn,numpy as np,pandas as pd,pickle
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets,preprocessing
from sklearn.model_selection import train_test_split,cross_val_score,validation_curve,learning_curve
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

      
p=pd.read_csv('city.txt',header=None,encoding='gb18030')    #pd读取csv数据

def f(cl):
    print(chr(11)*77)
    print('当前cl={}'.format(cl))
    data,cityname,data1=p.ix[:,1:],p.ix[:,0],p.ix[:,1:].sum(axis=1)     #data是数据，cityname是标签？
    km=KMeans(n_clusters=cl)              #创建kmeans对象
    label=km.fit_predict(data)            #训练data
    ex=np.sum(km.cluster_centers_,axis=1)       #对km.cluster_centers_横向求和
    cc=[]
    [cc.append([]) for i in range(ex.shape[0])]
    for i in range(len(cityname)):
        cc[label[i]].append(cityname[i])
    for i in range(len(cc)):
        print('ex=',ex[i])
        print(cc[i])
f(5)
