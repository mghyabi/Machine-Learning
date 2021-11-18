#a)
import numpy as np
import matplotlib.pyplot as plt

def concentGauss(N,vc,vo,r):
    c=np.sqrt(vc)*np.random.randn(2,N//2)
    l1=np.ones((1,N//2))
    E1=np.vstack((c,l1))

    ro=np.sqrt(vo)*np.random.randn(1,N//2)+r
    t=np.random.uniform(low=0,high=2*np.pi,size=N//2)
    l2=-1*np.ones((1,N//2))
    E2=np.vstack((ro*np.cos(t),ro*np.sin(t),l2))
    
    E=np.hstack((E1,E2))
    return E


N=500
r=5
vc=1
vo=1

C=concentGauss(N,vc,vo,r).T

fig=plt.figure(figsize=[10,10])
plt.scatter(C[C[:,2]==1,0], C[C[:,2]==1,1],s=80, alpha=0.8, c='blue', edgecolors='none', marker="+",label='target: +1')
plt.scatter(C[C[:,2]==-1,0], C[C[:,2]==-1,1],s=50, alpha=0.8, c='green', edgecolors='none', marker="x",label='target: -1')
plt.axis([-10,10,-10,10])
plt.grid('True',linestyle='--', linewidth=0.5)
plt.xlabel('$x_1$',fontsize=20)
plt.ylabel('$x_2$',fontsize=20)
plt.xticks(range(-10,11,2),fontsize=10)
plt.yticks(range(-10,11,2),fontsize=10)
plt.legend(fontsize=15)
plt.show()
# fig.savefig('p6_a.svg',format='svg')
# fig.savefig('p6_a.tif')

for K in [1,5,15]:
    nn=200
    nodes=np.linspace(-10,10,nn)
    x, y = np.meshgrid(nodes, nodes)
    NodeTag=np.zeros((nn,nn))

    for i in np.arange(nn):
        for j in np.arange(nn):
            a=np.tile([nodes[i],nodes[j]],(N,1))
            d=np.sqrt(np.reshape(np.sum((a-C[:,0:2])**2,axis=1),(N,1)))
            d=np.hstack((d,C[:,2,None]))
            d=d[d[:,0].argsort()]
            NodeTag[j,i]=np.sum(d[0:K,1])/np.abs(np.sum(d[0:K,1]))

    X=np.ndarray.flatten(x)
    Y=np.ndarray.flatten(y)
    Z=np.ndarray.flatten(NodeTag)

    fig=plt.figure(figsize=[10,10],dpi=300)
    plt.scatter(X[Z==1], Y[Z==1],s=0.5, alpha=0.8, c='blue', marker="+")
    plt.scatter(X[Z==-1], Y[Z==-1],s=0.5, alpha=0.8, c='green',marker="+")
    plt.xlabel('$\mathregular{x_1}$',fontsize=20)
    plt.ylabel('$\mathregular{x_2}$',fontsize=20)
    plt.xticks(np.arange(-10,11,5),fontsize=12)
    plt.yticks(np.arange(-10,11,5),fontsize=12)
    plt.title('K='+str(K),fontsize=20)
    plt.show()
#     fig.savefig('p6_k'+str(K)+'.svg',format='svg')
#     fig.savefig('p6_k'+str(K)+'.tif')
