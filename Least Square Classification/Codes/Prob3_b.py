#P3_b)
for name in dir():
    del globals()[name]
    
import numpy as np
import matplotlib.pyplot as plt
import os

ad=os.getcwd()
ad=ad+'\data\\'
X=np.loadtxt(ad+'DatasetA2_data.csv', unpack='True',delimiter=',').T
T=np.reshape(np.loadtxt(ad+'DatasetA2_labels.csv', unpack='True'),(X.shape[0],1))

#making hot key coding
T=np.tile(T,(1,int(np.max(T))+1))
for i in np.arange(T.shape[1]):
    T[:,i]-=(i+1)
T=np.equal(T,-1*np.ones((np.shape(T))))*1
T.astype(int)

#calcultating mean of each class
m=np.zeros((X.shape[1],T.shape[1]))
for i in np.arange(T.shape[1]):
    aa=list(map(bool,T[:,i]))
    m[:,i]=np.mean(X[aa,:],axis=0)

w=m[:,1]-m[:,0]
w/=np.linalg.norm(w)

y=X@w
y0=np.mean(y)-w[1]
C=(y>y0)*1
C=np.reshape(C,(X.shape[0],1))
#making hot key coding
C=np.tile(C,(1,int(np.max(C))+1))
for i in np.arange(C.shape[1]):
    C[:,i]-=(i+1)
C=np.equal(C,-1*np.ones((np.shape(C))))*1

#hyper planes
x=np.linspace(np.min(X[:,1]-1),np.max(X[:,1])+1,3)
y1=(w[0]*x-y0)/-w[1]
#indexing data according to lables
i0=np.squeeze(np.equal(T[:,0,None],np.ones((T.shape[0],1))))
i1=np.squeeze(np.equal(T[:,1,None],np.ones((T.shape[0],1))))

#plotting
fig=plt.figure(figsize=[10,10],dpi=300)
plt.scatter(X[i0,0], X[i0,1], alpha=0.8,marker='x', c='r', edgecolors='none', s=40)
plt.scatter(X[i1,0], X[i1,1], alpha=0.8,marker='o', facecolor='none', edgecolors='blue', s=50, linewidth=2)
plt.plot(x,y1,color='g')
plt.axis([-2,10,-7,5])
plt.grid('True',linestyle='--', linewidth=1)
plt.xlabel('$x_1$',fontsize=20)
plt.ylabel('$x_2$',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
# fig.savefig('p3_b1.svg',format='svg')


fig2=plt.figure(figsize=[10,10],dpi=300)
plt.hist(y[T[:,0]==1],7,alpha=0.3, align='mid',rwidth=0.8,facecolor='r',edgecolor='black',linewidth=2)
plt.hist(y[T[:,1]==1],15,alpha=0.3, align='mid',rwidth=0.8,facecolor='b',edgecolor='black',linewidth=2)
plt.xlabel('y',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
# fig2.savefig('p3_b2.svg',format='svg')

