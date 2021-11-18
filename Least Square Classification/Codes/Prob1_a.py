#P1_a)
for name in dir():
    del globals()[name]
    
import numpy as np
import matplotlib.pyplot as plt
import os

ad=os.getcwd()
ad=ad+'\data\\'
X=np.loadtxt(ad+'DatasetA_data.csv', unpack='True',delimiter=',').T
X=np.hstack((np.ones((X.shape[0],1)),X))
T=np.reshape(np.loadtxt(ad+'DatasetA_labels.csv', unpack='True'),(X.shape[0],1))

#making hot key coding
T=np.tile(T,(1,int(np.max(T))+1))
for i in np.arange(T.shape[1]):
    T[:,i]-=(i+1)
T=np.equal(T,-1*np.ones((np.shape(T))))*1

#W calculation
W=((np.linalg.inv(X.T@X))@X.T)@T

#classifyimg
y=X@W
C=np.equal(y,np.tile(np.amax(y,axis=1,keepdims=True),(1,T.shape[1])))*1

#printing the grid
nn=100
nodes=np.linspace(-2,7,nn)
x1, x2 = np.meshgrid(nodes, nodes)
NodeTag=np.zeros((nn,nn))
crd=np.stack((x1,x2),axis=2)
crd=np.reshape(crd,(10000,2),order='C')
crd=np.hstack((np.ones((crd.shape[0],1)),crd))

y2=crd@W
C2=np.equal(y2,np.tile(np.amax(y2,axis=1,keepdims=True),(1,T.shape[1])))*1

#hyper planes
x=np.linspace(np.min(X[:,1]-1),np.max(X[:,1])+1,3)
y1=(+0.5*W[0,0]-0.5*W[0,1]+W[1,0]*x)/-W[2,0]
#indexing data according to lables
i0=np.squeeze(np.equal(C[:,0,None],np.ones((C.shape[0],1))))
i1=np.squeeze(np.equal(C[:,1,None],np.ones((C.shape[0],1))))
i2=np.squeeze(np.equal(C2[:,0,None],np.ones((C2.shape[0],1))))
i3=np.squeeze(np.equal(C2[:,1,None],np.ones((C2.shape[0],1))))
#plotting
fig=plt.figure(figsize=[10,10],dpi=300)
plt.scatter(X[i0,1], X[i0,2], alpha=0.8,marker='x', c='r', edgecolors='none', s=40)
plt.scatter(X[i1,1], X[i1,2], alpha=0.8,marker='o', facecolor='none', edgecolors='blue', s=50, linewidth=2)
plt.scatter(crd[i2,1], crd[i2,2], alpha=0.8,marker='x', c='r', edgecolors='none', s=1)
plt.scatter(crd[i3,1], crd[i3,2], alpha=0.8,marker='o', facecolor='none', edgecolors='blue', s=1, linewidth=2)
plt.plot(x,y1,color='g')
plt.axis([-2,7,-2,7])
# plt.axis('equal')
plt.grid('True',linestyle='--', linewidth=1)
plt.xlabel('$x_1$',fontsize=20)
plt.ylabel('$x_2$',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
# fig.savefig('p1_a.svg',format='svg')
