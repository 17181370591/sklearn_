from sklearn import linear_model
import matplotlib.pyplot as plt,numpy as np
ll1=linear_model.Ridge(alpha=0.1)   #岭回归需要设置alpha
ll2=linear_model.LinearRegression()
x=np.array([0,1,2,3,4,5]).reshape(-1,1)
y=[0,1,2,3,4,6]
ls=(ll1,ll2)
plt.figure()
for i in range(len(ls)):
    ls[i].fit(x,y)
    print(ls[i].coef_,ls[i].intercept_)
    plt.subplot(221+i)
    plt.scatter(x,y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x,ls[i].predict(x),label='test')
    plt.legend('left upper')
plt.show()
s=(-10,-1,-0.1,-0.01,-0.001)
rcv=linear_model.RidgeCV(alphas=s)      #一次测试多个岭回归，返回效果最好的岭回归
rcv.fit(x,y)
print(rcv.alpha_)           #效果最好的岭回归的alpha
plt.scatter(x,y)
plt.plot(x,rcv.predict(x),label='test')
plt.show()
