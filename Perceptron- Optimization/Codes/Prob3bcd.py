#p3_bcd

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

def doublemoon(N,d,r,w):
    ro1=np.random.uniform(low=r-w/2,high=r+w/2,size=N//2)
    t1=np.random.uniform(low=0,high=np.pi,size=N//2)
    x1=ro1*np.cos(t1)
    y1=ro1*np.sin(t1)
    l1=np.ones((1,N//2))
    
    ro2=np.random.uniform(low=r-w/2,high=r+w/2,size=N//2)
    t2=np.random.uniform(low=np.pi,high=2*np.pi,size=N//2)
    x2=ro2*np.cos(t2)+r
    y2=ro2*np.sin(t2)-d
    l2=-1*np.ones((1,N//2))
    
    E=np.vstack((x1,y1,l1,x2,y2,l2))
    return E

start_time=time.time()

#parameter definition
eta=5e-6
N=500
r=1
w=0.6
d=0.0
NumIter=100
NumSample=30

#generating different datasets for each iteration
E=doublemoon(N,d,r,w)
E=E.T
E=np.vstack((E[:,0:3],E[:,3:6]))

X=np.hstack((np.ones((E.shape[0],1)),E[:,0:2]))
D=E[:,2,None]

J=np.zeros((NumSample,NumIter))
acc=np.zeros((NumSample,NumIter))
for j in np.arange(NumSample):
    #initializing W vector
    W=np.random.uniform(low=-0.5,high=0.5,size=X.shape[1])

    Iter=0
    while Iter<NumIter:
        y=perceptron(X,W)
        #finding misclassified data vectors
        i=np.squeeze(np.equal(np.abs(D-y),2*np.ones((y.shape[0],1))))
        #batch learning
        W += eta*np.sum(X[i,:]*D[i,:],axis=0)
        ww=np.reshape(W,(1,np.size(W)))
        J[j,Iter]=np.sum((ww@X.T)**2)
        ac=np.squeeze(np.equal(np.abs(D-y),np.zeros((y.shape[0],1))))*1
        acc[j,Iter]=np.sum(ac)/N
        Iter+=1

J=np.mean(J,axis=0)        
acc=np.mean(acc,axis=0)
        
fig1=plt.figure(figsize=[10,5])
plt.plot(np.arange(NumIter),J,c="b")
plt.xlabel('Iteration in the epoch',fontsize=20)
plt.ylabel('J',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid('True',linestyle='--', linewidth=1)
plt.show(fig1)
#fig1.savefig('p3_c1.svg',format='svg')

fig2=plt.figure(figsize=[10,5])
plt.plot(np.arange(NumIter),acc,c="b")
plt.xlabel('Iteration in the epoch',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.ylim(np.min(acc)/1.25,1.0)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid('True',linestyle='--', linewidth=1)
plt.show(fig2)
#fig2.savefig('p3_c2.svg',format='svg')

print("--- %s seconds ---" % (time.time() - start_time))
