#P3_c)
for name in dir():
    del globals()[name]
    
import numpy as np
import matplotlib.pyplot as plt
import os

def doublemoon(N,d,r,w):
    ro1=np.random.uniform(low=r-w/2,high=r+w/2,size=N//2)
    t1=np.random.uniform(low=0,high=np.pi,size=N//2)
    x1=ro1*np.cos(t1)
    y1=ro1*np.sin(t1)
    l1=np.zeros((1,N//2))
    
    ro2=np.random.uniform(low=r-w/2,high=r+w/2,size=N//2)
    t2=np.random.uniform(low=np.pi,high=2*np.pi,size=N//2)
    x2=ro2*np.cos(t2)+r
    y2=ro2*np.sin(t2)-d
    l2=np.ones((1,N//2))
    
    E=np.vstack((x1,y1,l1,x2,y2,l2))
    return E

N=5000
d=-0.1
r=1
w=0.6
E=doublemoon(N,d,r,w)
E=E.T

X=np.vstack((E[:,0:2],E[:,3:5]))
T=np.vstack((E[:,2,None],E[:,5,None]))

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
y0=np.mean(y)
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
plt.scatter(X[i0,0], X[i0,1], alpha=0.8,marker='x', c='r', edgecolors='none', s=8)
plt.scatter(X[i1,0], X[i1,1], alpha=0.8,marker='o', facecolor='none', edgecolors='blue', s=10, linewidth=1)
plt.plot(x,y1,color='g')
plt.axis([-1.5,2.5,-1.5,2.5])
plt.grid('True',linestyle='--', linewidth=1)
plt.xlabel('$x_1$',fontsize=20)
plt.ylabel('$x_2$',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
# fig.savefig('p3_c1.svg',format='svg')


fig2=plt.figure(figsize=[10,10],dpi=300)
plt.hist(y[T[:,0]==1],10,alpha=0.3, align='mid',rwidth=0.8,facecolor='r',edgecolor='black',linewidth=2)
plt.hist(y[T[:,1]==1],10,alpha=0.3, align='mid',rwidth=0.8,facecolor='b',edgecolor='black',linewidth=2)
plt.xlabel('y',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
# fig2.savefig('p3_c2.svg',format='svg')
