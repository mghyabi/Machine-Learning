#p2_a

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
eta=.001
N=500
r=1
w=0.6
d=-0.5

#iterating 30 times for a single epoch
J=np.zeros((30,N))
acc=np.zeros((30,N))
for k in np.arange(30):
    #generating different datasets for each iteration
    E=doublemoon(N,d,r,w)
    E=E.T
    E=np.vstack((E[:,0:3],E[:,3:6]))

    X=np.hstack((np.ones((E.shape[0],1)),E[:,0:2]))
    D=E[:,2,None]

    #initializing W vector
    W=np.zeros((1,X.shape[1]))

    #shuffling
    i=np.arange(N)
    np.random.shuffle(i)
    X=X[i,:]
    D=D[i,:]
    
    #epoch
    for i in np.arange(N):
        #updating W vector
        y=perceptron(X[None,i,:],W)
        e=D[i,:]-y
        W=W+eta*e*X[i,:]
        
        #measuring the cost with each updated W vector
        y=perceptron(X,W)
        J[k,i]=np.sum(0.5*(y-D)**2)
        ac=np.squeeze(np.equal(np.abs(D-y),np.zeros((y.shape[0],1))))*1
        acc[k,i]=np.sum(ac)/N

J=np.mean(J,axis=0)        
acc=np.mean(acc,axis=0)

fig1=plt.figure(figsize=[10,5])
plt.plot(np.arange(N),J,c="b")
plt.xlabel('Iteration in the epoch',fontsize=20)
plt.ylabel('J',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid('True',linestyle='--', linewidth=1)
plt.show(fig1)
# fig1.savefig('p2_a1_-5.svg',format='svg')

fig3=plt.figure(figsize=[10,5])
plt.plot(np.arange(N),acc,c="b")
plt.xlabel('Iteration in the epoch',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.ylim(np.min(acc)/1.25,1.05)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid('True',linestyle='--', linewidth=1)
plt.show(fig3)
# fig3.savefig('p2_a2_-5.svg',format='svg')

print("--- %s seconds ---" % (time.time() - start_time))
#_______________________________________________________________________#
#p2_b

#printing the grid
nn=100
nodes1=np.linspace(-1.5,2.5,nn)
nodes2=np.linspace(-2.0,2.0,nn)
x1, x2 = np.meshgrid(nodes1, nodes2)
NodeTag=np.zeros((nn,nn))
crd=np.stack((x1,x2),axis=2)
crd=np.reshape(crd,(10000,2),order='C')
crd=np.hstack((np.ones((crd.shape[0],1)),crd))
y1=perceptron(crd,W)
i4=np.squeeze(np.equal(y1,np.ones((y1.shape[0],1))))
i5=np.squeeze(np.equal(y1,-1*np.ones((y1.shape[0],1))))
#hyper planes
x=np.linspace(-5,5,3)
y1=(W[0,0]+W[0,1]*x)/-W[0,2]
#indexing data according to lables
i1=np.squeeze(np.equal(y+D,2*np.ones((y.shape[0],1))))
i2=np.squeeze(np.equal(y+D,-2*np.ones((y.shape[0],1))))
i3=np.squeeze(np.equal(y+D,np.zeros((y.shape[0],1))))
#plotting
fig2=plt.figure(figsize=[10,10],dpi=300)
plt.scatter(X[i1,1], X[i1,2], alpha=0.8,marker='+', c='b', edgecolors='none', s=75)
plt.scatter(X[i2,1], X[i2,2], alpha=0.8,marker='x', c='g', edgecolors='none', s=55)
plt.scatter(X[i3,1], X[i3,2], alpha=0.8,marker='*', c='r', edgecolors='none', s=95)
plt.scatter(crd[i4,1], crd[i4,2], alpha=0.8,marker='+', c='b', edgecolors='none', s=3)
plt.scatter(crd[i5,1], crd[i5,2], alpha=0.8,marker='x', c='g', edgecolors='none', s=3)
plt.plot(x,y1,color='r')
plt.axis([-1.5,2.5,-2.0,2.0])
plt.grid('True',linestyle='--', linewidth=1)
plt.xlabel('$x_1$',fontsize=20)
plt.ylabel('$x_2$',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show(fig2)
# fig2.savefig('p2_b_-5.svg',format='svg')
