#a, b and c)
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio

def Batch_K_Means(E,K):
    D=np.shape(E)
    N=D[1]
    D=D[0]
    j=1
    
    mu_index=np.random.randint(N,size=K)
    mu_old=E[:,mu_index]
    d=np.zeros((K,N))
    
    while j<10000:#max iterations
        #distances (d) d has K rows (one for each cluster) and N columns (one for each datapoint)
        for i in np.arange(K):
            d[i,:]=np.sum((np.tile(np.reshape(mu_old[:,i],(D,1),order='C'),(1,N))-E)**2,axis=0)

        #using boolean operators to determine r from d
        r=np.equal(d,np.tile(np.amin(d,axis=0),(K,1)))*1

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

K=4 #number of clusters
name="nature-1.png"
#import image from data folder
ad=os.getcwd()
ad=ad+'\data\\'
I=imageio.imread(ad+name)
#remove the transparency layer of png
I=np.dstack((I[:,:,0],I[:,:,1],I[:,:,2]))
s=np.shape(I)


R=I[:,:,0]
G=I[:,:,1]
B=I[:,:,2]
R=np.reshape(R,(1,s[0]*s[1]),order='C')
G=np.reshape(G,(1,s[0]*s[1]),order='C')
B=np.reshape(B,(1,s[0]*s[1]),order='C')
R=np.squeeze(R)
G=np.squeeze(G)
B=np.squeeze(B)

E=np.vstack((R,G,B))

fig=plt.figure(figsize=[10,10],dpi=300)
plt.imshow(I)
plt.axis('off')

Kmeans=Batch_K_Means(E,K)
mu=Kmeans[0]/255
r=Kmeans[3]
r=r*np.tile(np.arange(K)+1,(s[0]*s[1],1)).T
r=np.sum(r,axis=0)

R=mu[0,r-1]
G=mu[1,r-1]
B=mu[2,r-1]
R=np.reshape(R,(s[0],s[1]),order='C')
G=np.reshape(G,(s[0],s[1]),order='C')
B=np.reshape(B,(s[0],s[1]),order='C')
I_new=np.dstack((R,G,B))

fig=plt.figure(figsize=[10,10],dpi=300)
plt.imshow(I_new)
plt.axis('off')
# fig.savefig('p3_ck4.svg',format='svg')
# fig.savefig('p3_ck4.tif')
