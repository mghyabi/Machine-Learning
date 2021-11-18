#b)
import numpy as np
import matplotlib.pyplot as plt

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
    
    E=np.hstack((np.vstack((x1,y1,l1)),np.vstack((x2,y2,l2))))
    return E


N=500
d=0
r=1
w=0.6

C=doublemoon(N,d,r,w).T

fig=plt.figure(figsize=[10,10])
plt.scatter(C[C[:,2]==1,0], C[C[:,2]==1,1],s=80, alpha=0.8, c='blue', edgecolors='none', marker="+",label='target: +1')
plt.scatter(C[C[:,2]==-1,0], C[C[:,2]==-1,1],s=50, alpha=0.8, c='green', edgecolors='none', marker="x",label='target: -1')
plt.axis([-3,3,-3,3])
plt.grid('True',linestyle='--', linewidth=0.5)
plt.xlabel('$x_1$',fontsize=20)
plt.ylabel('$x_2$',fontsize=20)
plt.xticks(range(-3,4),fontsize=15)
plt.yticks(range(-3,4),fontsize=15)
plt.legend(fontsize=15)
plt.show()
# fig.savefig('p6_b.svg',format='svg')
# fig.savefig('p6_b.tif')

for K in [1,5,15]:
    nn=200
    nodes=np.linspace(-3,3,nn)
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
    plt.axis([-3,3,-3,3])
    plt.scatter(X[Z==1], Y[Z==1],s=0.5, alpha=0.8, c='blue', marker="+")
    plt.scatter(X[Z==-1], Y[Z==-1],s=0.5, alpha=0.8, c='green',marker="+")
    plt.xlabel('$\mathregular{x_1}$',fontsize=20)
    plt.ylabel('$\mathregular{x_2}$',fontsize=20)
    plt.xticks(range(-3,4),fontsize=15)
    plt.yticks(range(-3,4),fontsize=15)
    plt.title('K='+str(K),fontsize=20)
    plt.show()
#     fig.savefig('p6_b_k'+str(K)+'.svg',format='svg')
#     fig.savefig('p6_b_k'+str(K)+'.tif')
