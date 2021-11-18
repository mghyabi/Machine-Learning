import numpy as np
import matplotlib.pyplot as plt
import time

def circGauss(N,mu,var):
    D=np.size(mu)
    mu=np.reshape(mu,(D,1),order='C')
    E=np.sqrt(var)*np.random.randn(D,N)+mu
    return E

def Kmeans(E,K,eta,tol):
    D=E.shape[0]
    N=E.shape[1]
    #initializing mu by shuffling data points and d
    i=np.arange(N)
    np.random.shuffle(i)
    E=E[:,i]

    mu=E[:,0:K]
    mu_old=mu.copy()
    d=np.zeros((K,N))

    l=1
    while l<10000:
        #distances (d) d has K rows (one for each cluster) and N columns (one for each datapoint)
        for i in np.arange(K):
            d[i,:]=np.sum((np.tile(np.reshape(mu[:,i],(D,1),order='C'),(1,N))-E)**2,axis=0)

        #defining tags (r) and index for the mu to change at each data point(r2)
        r=np.equal(d,np.tile(np.amin(d,axis=0),(K,1)))*1
        r2=r*(np.tile(np.arange(D).reshape((D,1)),(1,N))+1)

        #Learning algorithm
        for j in np.arange(N):
            k=np.sum(r2[:,j])
            mu[:,k-1]=mu[:,k-1]+eta*(E[:,j]-mu[:,k-1])
        if np.allclose(mu_old,mu,atol=tol):
            break
        #initializing mu by shuffling data points and d
        i=np.arange(N)
        np.random.shuffle(i)
        E=E[:,i]
        l+=1
        mu_old=mu.copy()
    return mu,l

start_time=time.time()

NSample=10
etas=np.arange(1e-4,1.6e-3,1e-4)
Reps=np.zeros((NSample,etas.shape[0]))

for i in np.arange(etas.shape[0]):
    for j in np.arange(Reps.shape[0]):
        eta=etas[i] #learning rate
        tol=1e-3 #convergence criteria
        N=500 #number of data points
        K=2 #number of clusters
        E1=circGauss(N//2,[0,0],3)
        E2=circGauss(N//2,[5,5],3)
        E=np.hstack((E1,E2))

        Result=Kmeans(E,K,eta,tol)
        Reps[j,i]=Result[1]
Reps=np.mean(Reps,axis=0)

print("--- %s seconds ---" % (time.time() - start_time))

fig=plt.figure(figsize=[10,5])
plt.plot(etas,Reps,c="b")
plt.xlabel('Learning Rate ($\eta$)',fontsize=20)
plt.ylabel('Average Number of Iterations',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid('True',linestyle='--', linewidth=1)
plt.show()
#fig.savefig('p2.svg',format='svg')
#fig.savefig('p2.tif')
