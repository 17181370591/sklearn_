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

def load_dataset(feature_paths,label_paths):
    feature=np.ndarray(shape=(0,41))
    label=np.ndarray(shape=(0,1))
    for file in feature_paths:
        df=pd.read_table(file,delimiter=',',na_values='?',header=None)
        imp=Imputer(missing_values='NaN',strategy='mean',axis=0)        #填充none值
        imp.fit(df)
        df=imp.transform(df)
        feature=np.concatenate((feature,df))                #concatenate是合并，可以设置axis
    for file in label_paths:
        df=pd.read_table(file,header=None)
        label=np.concatenate((label,df))
    label=np.ravel(label)                                   #将label降成1维
    return feature,label
if __name__=='__main__':
    feature_paths=[r'A.feature',r'B.feature',r'C.feature',r'D.feature',r'E.featurE']
    label_paths=[r'A.label',r'B.label',r'C.label',r'D.label',r'E.label']
    x_train,y_train=load_dataset([feature_paths[0]],[label_paths[0]])
    x_test,y_test=load_dataset([feature_paths[1]],[label_paths[1]])
    x_train,x_,y_train,y_=train_test_split(x_train,y_train,test_size=0.0)       #由于test_size=0，所以x_，y_都是None，作用是乱序
    print('start traing knn')
    knn=KNeighborsClassifier().fit(x_train,y_train)
    a_knn=knn.predict(x_test)
    print('start traing dt')
    dt=DecisionTreeClassifier().fit(x_train,y_train)
    a_dt=dt.predict(x_test)
    print('start traing gusaanb')
    gnb=GaussianNB().fit(x_train,y_train)
    a_gnb=gnb.predict(x_test)
    print('test knn')
    print(classification_report(y_test,a_knn))
    print('test db')
    print(classification_report(y_test,a_dt))
    print('test guassnb')
    print(classification_report(y_test,a_gnb))
