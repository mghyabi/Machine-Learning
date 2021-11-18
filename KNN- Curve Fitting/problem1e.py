import numpy as np
import matplotlib.pyplot as plt

def circGauss(N,mu,var):
    D=np.size(mu)
    mu=np.reshape(mu,(D,1),order='C')
    E=np.sqrt(var)*np.random.randn(D,N)+mu
    return E

#e)
N=50 #number of data points
D=2 #number of dimensions
K=5 #number of clusters
E1=circGauss(N//2,[0,0],3)
E2=circGauss(N//2,[5,5],3)
E=np.hstack((E1,E2))

for i in np.arange(3):
    mu_index=np.random.randint(N,size=K)
    mu_old=E[:,mu_index]
    d=np.zeros((K,N))
    mup=mu_old.copy()

    j=1
    while j<50000:
        #distances (d) d has K rows (one for each cluster) and N columns (one for each datapoint)
        for j in np.arange(K):
            d[j,:]=np.sum((np.tile(np.reshape(mu_old[:,j],(D,1),order='C'),(1,N))-E)**2,axis=0)

        #using boolean operators to determine r from d
        r=np.equal(d,np.tile(np.amin(d,axis=0),(K,1)))*1

        #updating mu for the first time
        mu_new=np.zeros((K,D))
        for j in np.arange(K):
            num=np.sum(np.tile(r[j,:],(D,1))*E,axis=1)
            den=np.sum(np.tile(r[j,:],(D,1)),axis=1)
            mu_new[j,:]=np.squeeze(num/den)

        mu_new=mu_new.T

        if np.allclose(mu_old,mu_new):
            break
        mu_old=mu_new.copy()
        j+=1

    J=np.zeros((K,N))    
    for j in np.arange(K):
        J[j,:]=np.sum((np.tile(np.reshape(mu_new[:,j],(D,1),order='C'),(1,N))-E)**2,axis=0)

    J=np.sum(J*r)/N

    fig=plt.figure(figsize=[10,10],dpi=300)
    plt.scatter(E[0,:], E[1,:], alpha=0.8, c='blue', edgecolors='none', s=25,label='Data points')
    plt.scatter(mu_new[0,:], mu_new[1,:], alpha=0.8, c='red', edgecolors='none', s=270,label='Cluster centers',marker='*')
    plt.scatter(mup[0,:], mup[1,:], c='green', edgecolors='none', s=120,label='Initial centers',marker='s')
    for j in np.arange(K):
        plt.arrow(mup[0,j],mup[1,j],mu_new[0,j]-mup[0,j], mu_new[1,j]-mup[1,j],length_includes_head=True,head_width=.2, head_length=.3)
    plt.axis([-5,10,-5,10])
    plt.grid('True',linestyle='--', linewidth=1)
    plt.xlabel('$x_1$',fontsize=20)
    plt.ylabel('$x_2$',fontsize=20)
    plt.xticks([-5,0,5,10],fontsize=15)
    plt.yticks([-5,0,5,10],fontsize=15)
    plt.legend(fontsize=15)
    plt.show()
#     fig.savefig('p1_e' +str(i)+ '.svg',format='svg')
#     fig.savefig('p1_e' +str(i)+ '.tif')
