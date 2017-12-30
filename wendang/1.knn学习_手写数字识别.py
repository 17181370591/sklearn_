import matplotlib.pyplot as plt,pandas as pd,numpy as np,os
from sklearn.neighbors import KNeighborsClassifier as KNN



path1=r'C:\Users\Administrator\Desktop\trainingDigits\\'
path2=r'C:\Users\Administrator\Desktop\testDigits\\'
name1=list(os.walk(path1))[0][2]
s_all=[]
n_all=[]
name2=list(os.walk(path2))[0][2]
s_all2=[]
n_all2=[]
def f(nn,path1,l1,l2):
#nn='0_0.txt'
    n1=nn.split('_')[0]
    nn=path1+nn
    #p1=np.array(pd.read_csv(nn,header=None)).sum()
    with open(nn,'r') as f:
        lines=f.readlines()
        s=''.join(lines).replace('\n','')
        s=[i for i in s]
        #print(s)
    l1.append(s)
    l2.append(n1)
for i in name1:
    f(i,path1,s_all,n_all)
for i in name2:
    f(i,path2,s_all2,n_all2)

knn=KNN(n_neighbors=2)
knn.fit(s_all,n_all)
sc=knn.score(s_all2,n_all2)
