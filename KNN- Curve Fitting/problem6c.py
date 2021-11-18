#c)
import numpy as np
import matplotlib.pyplot as plt

def gaussX(N,v):
    E1=np.sqrt(v)*np.random.randn(2,N//2)
    E2=np.sqrt(v)*np.random.randn(2,N//2)
    #using sign to remove unwanted datapoints
    i=E1[0,:]*E1[1,:]
    E1=E1[:,i<=0]
    l1=np.ones((1,E1.shape[1]))
    E1=np.vstack((E1,l1))
    i=E2[0,:]*E2[1,:]
    E2=E2[:,i>=0]
    l2=-1*np.ones((1,E2.shape[1]))
    E2=np.vstack((E2,l2))
    E=np.hstack((E1,E2))
    return E

N=500
v=1

C=gaussX(N,v).T

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
# fig.savefig('p6_c.svg',format='svg')
# fig.savefig('p6_c.tif')

for K in [1,5,15]:
    nn=200
    nodes=np.linspace(-4,4,nn)
    x, y = np.meshgrid(nodes, nodes)
    NodeTag=np.zeros((nn,nn))

    for i in np.arange(nn):
        for j in np.arange(nn):
            a=np.tile([nodes[i],nodes[j]],(C.shape[0],1))
            d=np.sqrt(np.reshape(np.sum((a-C[:,0:2])**2,axis=1),(C.shape[0],1)))
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
#     fig.savefig('p6_c_k'+str(K)+'.svg',format='svg')
#     fig.savefig('p6_c_k'+str(K)+'.tif')
