#p3_e

for name in dir():
    del globals()[name]
    
import numpy as np
import matplotlib.pyplot as plt
import time

def perceptron(x,w):
    m=x.shape[0]
    n=x.shape[1]
    x=x.T
    if n>w.size:
        x=x.T
      
    n=w.size
    w=np.reshape(w,(1,n))
    v=w@x
    v=np.reshape(v,(m,1))
    y=np.zeros((m,1))
    y[v<0]=-1
    y[v>0]=1
        
    return y

def gaussX(N,v):
    E1=np.sqrt(v)*np.random.randn(2,N//2)
    E2=np.sqrt(v)*np.random.randn(2,N//2)
    i=E1[0,:]*E1[1,:]
    E1=E1[:,i<0]
    l1=np.ones((1,E1.shape[1]))
    E1=np.vstack((E1,l1))
    i=E2[0,:]*E2[1,:]
    E2=E2[:,i>0]
    l2=-1*np.ones((1,E2.shape[1]))
    E2=np.vstack((E2,l2))
    E=np.hstack((E1,E2))
    return E

start_time=time.time()

#parameter definition
eta=5e-6
N=500
v=1

#generating different datasets for each iteration
E=gaussX(N,v)
E=E.T

X=np.hstack((np.ones((E.shape[0],1)),E[:,0:2]))
D=E[:,2,None]

#initializing W vector
W=np.random.uniform(low=-0.5,high=0.5,size=X.shape[1])

Iter=0
while Iter<100:
    y=perceptron(X,W)
    #finding misclassified data vectors
    i=np.squeeze(np.equal(np.abs(D-y),2*np.ones((y.shape[0],1))))
    #batch learning
    W += eta*np.sum(X[i,:]*D[i,:],axis=0)
    Iter+=1
    
#printing the grid
nn=100
nodes=np.linspace(-4.0,4.0,nn)
x1, x2 = np.meshgrid(nodes, nodes)
NodeTag=np.zeros((nn,nn))
crd=np.stack((x1,x2),axis=2)
crd=np.reshape(crd,(10000,2),order='C')
crd=np.hstack((np.ones((crd.shape[0],1)),crd))
y1=perceptron(crd,W)
i4=np.squeeze(np.equal(y1,np.ones((y1.shape[0],1))))
i5=np.squeeze(np.equal(y1,-1*np.ones((y1.shape[0],1))))
#hyper planes
x=np.linspace(-5,5,3)
y1=(W[0]+W[1]*x)/-W[2]
#indexing data according to lables
i1=np.squeeze(np.equal(y+D,2*np.ones((y.shape[0],1))))
i2=np.squeeze(np.equal(y+D,-2*np.ones((y.shape[0],1))))
i3=np.squeeze(np.equal(y+D,np.zeros((y.shape[0],1))))
#plotting
fig2=plt.figure(figsize=[10,10],dpi=300)
plt.scatter(X[i1,1], X[i1,2], alpha=0.8,marker='+', c='b', edgecolors='none', s=75)
plt.scatter(X[i2,1], X[i2,2], alpha=0.8,marker='x', c='g', edgecolors='none', s=65)
plt.scatter(X[i3,1], X[i3,2], alpha=0.8,marker='*', c='r', edgecolors='none', s=105)
plt.scatter(crd[i4,1], crd[i4,2], alpha=0.8,marker='+', c='b', edgecolors='none', s=3)
plt.scatter(crd[i5,1], crd[i5,2], alpha=0.8,marker='x', c='g', edgecolors='none', s=3)
plt.plot(x,y1,color='r')
plt.axis([-4.0,4.0,-4.0,4.0])
plt.grid('True',linestyle='--', linewidth=1)
plt.xlabel('$x_1$',fontsize=20)
plt.ylabel('$x_2$',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show(fig2)
# fig2.savefig('p3_e.svg',format='svg')

print("--- %s seconds ---" % (time.time() - start_time))
print(W)
