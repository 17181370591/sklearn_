import matplotlib.pyplot as plt,pandas as pd,numpy as np,os,sklearn
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LinearRegression,Ridge
from sklearn import datasets,tree
from sklearn.model_selection  import train_test_split

iris=datasets.load_iris()
clf=tree.DecisionTreeClassifier()
x1,x2,y1,y2=train_test_split(iris.data,iris.target,test_size=0.2)
clf.fit(x1,y1)
#clf.fit(iris.data,iris.target)
cs=clf.score(x2,y2)
print(cs)
