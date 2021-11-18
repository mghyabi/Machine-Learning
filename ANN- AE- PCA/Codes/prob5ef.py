#P5_e
for name in dir():
    del globals()[name]
    
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

def Batch_K_Means(E,K):
    D=np.shape(E)
    N=D[1]
    D=D[0]
    j=1
    
    #initializing mu
    mu_index=np.random.randint(N,size=K)
    mu_old=E[:,mu_index]
    d=np.zeros((K,N))

    while j<10000:#max iterations
        #distances (d) d has K rows (one for each cluster) and N columns (one for each datapoint)
        for i in np.arange(K):
            d[i,:]=np.sum((np.tile(np.reshape(mu_old[:,i],(D,1),order='C'),(1,N))-E)**2,axis=0)

        #using boolean operators to determine r from d
        r=np.equal(d,np.tile(np.amin(d,axis=0),(K,1)))*1
        
        #checking at least one data point falls into each cluster
        check=np.prod(np.sum(r,axis=1))
        if check==0:
            while check==0:
                mu_index=np.random.randint(N,size=K)
                mu_old=E[:,mu_index]
                for i in np.arange(K):
                    d[i,:]=np.sum((np.tile(np.reshape(mu_old[:,i],(D,1),order='C'),(1,N))-E)**2,axis=0)
                r=np.equal(d,np.tile(np.amin(d,axis=0),(K,1)))*1
                check=np.prod(np.sum(r,axis=1))

        #updating mu 
        mu_new=np.zeros((K,D))
        for i in np.arange(K):
            num=np.sum(np.tile(r[i,:],(D,1))*E,axis=1)
            den=np.sum(np.tile(r[i,:],(D,1)),axis=1)
            mu_new[i,:]=np.squeeze(num/den)
        mu_new=mu_new.T
    
        if np.allclose(mu_old,mu_new):
            break
        mu_old=mu_new.copy()
        j+=1
    
    J=np.zeros((K,N))    
    for i in np.arange(K):
        J[i,:]=np.sum((np.tile(np.reshape(mu_new[:,i],(D,1),order='C'),(1,N))-E)**2,axis=0)

    J=np.sum(J*r)
    for i in np.arange(K):
        d[i,:]=np.sum((np.tile(np.reshape(mu_new[:,i],(D,1),order='C'),(1,N))-E)**2,axis=0)
    r=np.equal(d,np.tile(np.amin(d,axis=0),(K,1)))*1
    return mu_new, J, j, r

#import images from data folder
ad=os.getcwd()
ad=ad+'\data\\'

X=np.loadtxt(open(ad+"spikes.csv", "rb"), delimiter=",")    

pca = PCA(n_components=2)
PC2D = pca.fit_transform(X)
PC2D=PC2D.T

D=PC2D.shape[0] #number of dimensions

K=3 #number of clusters

Result=Batch_K_Means(PC2D,K)
mu=Result[0]
r=Result[3]

fig=plt.figure(figsize=[10,10],dpi=300)
plt.scatter(PC2D[0,r[0,:]==1], PC2D[1,r[0,:]==1], alpha=0.8, c='blue', edgecolors='none', s=30,label='Cluster 1')
plt.scatter(PC2D[0,r[1,:]==1], PC2D[1,r[1,:]==1], alpha=0.8, c='red', edgecolors='none', s=30,label='Cluster 2')
plt.scatter(PC2D[0,r[2,:]==1], PC2D[1,r[2,:]==1], alpha=0.8, c='green', edgecolors='none', s=30,label='Cluster 3')
plt.legend(fontsize=15)
plt.axis([-1.5e-4,2e-4,-1.5e-4,2e-4])
plt.grid('True',linestyle='--', linewidth=1)
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.xlabel('$x_1$',fontsize=20)
plt.ylabel('$x_2$',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
# fig.savefig(' p5_e.svg',format='svg')

#____________________
#P5_f

fig=plt.figure(figsize=[10,10])
for i in np.arange(PC2D.shape[1]):
    if r[0,i]==1:
        plt.plot(np.arange(1,27),X[i,:])
# plt.axis([0,30,-1,2])
plt.xlabel('t',fontsize=20)
plt.ylabel('Amplitude',fontsize=20)
plt.ticklabel_format(style='sci', axis='Y', scilimits=(0,0))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
# fig.savefig(' p5_f1.svg',format='svg')

fig=plt.figure(figsize=[10,10])
for i in np.arange(PC2D.shape[1]):
    if r[1,i]==1:
        plt.plot(np.arange(1,27),X[i,:])
# plt.axis([0,30,-1,2])
plt.xlabel('t',fontsize=20)
plt.ylabel('Amplitude',fontsize=20)
plt.ticklabel_format(style='sci', axis='Y', scilimits=(0,0))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
# fig.savefig(' p5_f2.svg',format='svg')

fig=plt.figure(figsize=[10,10])
for i in np.arange(PC2D.shape[1]):
    if r[2,i]==1:
        plt.plot(np.arange(1,27),X[i,:])
# plt.axis([0,30,-1,2])
plt.xlabel('t',fontsize=20)
plt.ylabel('Amplitude',fontsize=20)
plt.ticklabel_format(style='sci', axis='Y', scilimits=(0,0))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
# fig.savefig(' p5_f3.svg',format='svg')
