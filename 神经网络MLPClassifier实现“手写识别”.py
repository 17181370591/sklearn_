import sklearn,numpy as np,pandas as pd,pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets,preprocessing,metrics,linear_model
from sklearn.model_selection import train_test_split,cross_val_score,validation_curve,learning_curve
import numpy as np,pandas as pd,matplotlib.pyplot as plt,PIL
import matplotlib.pyplot as plt,os,time
from sklearn.cluster import DBSCAN,KMeans
from sklearn.decomposition import PCA,NMF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import Imputer   #导入预处理模块
from sklearn.metrics import classification_report         #导入预测结果评估模块
from sklearn.linear_model import Ridge          #调用岭回归模型
from sklearn.neural_network import MLPClassifier
f1=r'C:\Users\Administrator\Desktop\testDigits\0_0.txt'

#使用手写数字识别的数据
def img2vector(fn):
    fr=pd.read_csv(fn,header=None)
    return fr[0].map(lambda x:[i for i in x]).sum()
def img2vector1(fn):
    retMat=np.zeros([1024],int)
    fr=open(fn)
    lines=fr.readlines()
    for i in range(32):
        for j in range(32):
            retMat[i*32+j]=lines[i][j]
    return retMat    
#上方两个函数作用一样，读取fn的数据（32*32的数字），拼接成1024的np.array并返回

def readDataSet(path):
    fileList=os.listdir(path)       #查找path下所有文件/文件夹（只有文件）
    numFiles=len(fileList)          #返回文件数量
    dataSet=np.zeros([numFiles,1024],int)       #dataSet用来保存所有数字对应数据的img2vector生成的data
    hwLabels=np.zeros([numFiles,10])        #hwLabels用来保存所有数字，10表示用标签保存信息
    for i in range(numFiles):
        filePath=fileList[i]            #读取每个文件的名字
        digit=int(filePath.split('_')[0])   #文件名是a_b，返回a
        hwLabels[i][digit]=1
            #digit是第i个数据的数字（或者说标签），设置hwLabels第i个值的第digit个值为1，其他值不变（为0）
        dataSet[i]=img2vector(path+r'/'+filePath)
            #dataSet[i]的值是path+r'/'+filePath（完整名字）数据返回的1024的np.array
    return dataSet,hwLabels

t1=time.clock()
train_dataSet,train_hwLabels=readDataSet('trainingDigits')  #读取用作train的所有数据
clf=(hidden_layer_sizes=(100,),activation='logistic',
                  solver='adam',learning_rate_init=0.01,max_iter=500)
            #创建MLPClassifier，参数完全看不懂？？？
clf.fit(train_dataSet,train_hwLabels)       #训练

dataSet,hwLabels=readDataSet('testDigits')      #读取用作test的所有数据

res=clf.predict(dataSet)            #返回预测的结果
error_num=0                         #设置初始错误值=0
num=len(dataSet)                    
for i in range(num):                #遍历所有预测的结果
    if np.sum(res[i]==hwLabels[i])<10:      #如果正确np.sum(res[i]==hwLabels[i])=10（10个True），否则=8
        error_num+=1                
print(time.clock()-t1)
print('num=',num)
print('error_num',error_num)
print('error_num/num=',error_num/num)   
