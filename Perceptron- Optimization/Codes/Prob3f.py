#p3_f

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
NumIter=100
NumSample=30

#generating different datasets for each iteration
E=gaussX(N,v)
E=E.T

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
plt.xlabel('Iteration',fontsize=20)
plt.ylabel('J',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid('True',linestyle='--', linewidth=1)
plt.show(fig1)
# fig1.savefig('p3_f1.svg',format='svg')

fig2=plt.figure(figsize=[10,5])
plt.plot(np.arange(NumIter),acc,c="b")
plt.xlabel('Iteration',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.ylim(np.min(acc)/1.25,1.0)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid('True',linestyle='--', linewidth=1)
plt.show(fig2)
# fig2.savefig('p3_f2.svg',format='svg')

print("--- %s seconds ---" % (time.time() - start_time))
