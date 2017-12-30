import matplotlib.pyplot as plt,pandas as pd,numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split

def f(l):   #返回最大值最小值的差，用每个值去除以差，进行数据归一化即[0,1）
    return l.max()-l.min()

p1=pd.read_csv('1.csv',header=None)
xx=p1.ix[:,0:2]                 #xx是3组的特征
p2=np.array((f(xx[0]),f(xx[1]),f(xx[2])))
x=xx/p2                         #归一化
yy=p1.ix[:,3]                   #target

x1,x2,y1,y2=train_test_split(x,yy,test_size=0.1)
knn=KNN(n_neighbors=3)
knn.fit(x1,y1)
print(knn.score(x2,y2))


fig=plt.figure()        #创建画布
ax=fig.add_subplot(321)         #在3*2的第1个位置创建子图
ax.scatter(x=x.ix[:,0],y=x.ix[:,1],s=pd.Series(yy)*11,c=pd.Series(yy))
#重要！！！
#散点图，这里s和c的参数是与x,y相同尺寸的内容是value=[1,2,3]的数组，
#函数会对相同的value给予相同的size和color
plt.show()
