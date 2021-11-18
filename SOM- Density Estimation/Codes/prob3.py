#P3
for name in dir():
    del globals()[name]
    
import numpy as np
import matplotlib.pyplot as plt
import os

def Histogram(X,N):
    b=np.linspace(np.min(X),np.max(X),num=N)
    h=np.zeros(N)
    for m in np.arange(len(X)):
        e=np.abs(b-X[m])
        e=np.where(e==np.min(e),1,0)
        h+=e
    return h,b

#import from data folder
ad=os.getcwd()
ad=ad+'\data\\'
X=np.loadtxt(open(ad+"DatasetA.csv", "rb"), delimiter=",")

Nb=2000
data=Histogram(X,Nb)
h=data[0]
x=data[1]

fig1=plt.figure(figsize=[20,10],dpi=300)
plt.bar(x,h,width=50/Nb,facecolor='r',edgecolor=None,linewidth=0.5)
plt.ylabel('Histogram',fontsize=20)
plt.xlabel('x',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show(fig1)
# fig1.savefig('P3_5.svg',format='svg')


K=3

#initialization
pi=np.tile(np.ones(K)/K,(Nb,1))
sigma=np.tile(np.ones(K),(Nb,1))
mu=np.tile(np.random.choice(x,size=K,replace=False),(Nb,1))
x=np.tile(x,(K,1)).T
p=pi*(1/np.sqrt(2*np.pi*sigma))*np.exp(-(x-mu)**2/(2*sigma))
p/=np.tile(np.sum(p,axis=1,keepdims=True),(1,K))
ll=np.sum(np.log(np.sum(p,axis=1)))

Iter=1
while Iter<100:
    #updating
    mu=np.sum(p*x,axis=0)/np.sum(p,axis=0)
    sigma=np.sum(p*(x-mu)**2,axis=0)/np.sum(p,axis=0)
    pi=np.sum(p,axis=0)/Nb

    pi=np.tile(pi,(Nb,1))
    sigma=np.tile(sigma,(Nb,1))
    mu=np.tile(mu,(Nb,1))
    p=pi*(1/np.sqrt(2*np.pi*sigma))*np.exp(-(x-mu)**2/(2*sigma))
    p/=np.tile(np.sum(p,axis=1,keepdims=True),(1,K))

    ll_new=np.sum(np.log(np.sum(p,axis=1)))
    if (ll_new-ll)/ll_new<1e-5:
        break
    ll=ll_new
    Iter+=1

mu=np.sum(p*x,axis=0)/np.sum(p,axis=0)
sigma=np.sum(p*(x-mu)**2,axis=0)/np.sum(p,axis=0)
pi=np.sum(p,axis=0)/Nb
p=(1/np.sqrt(2*np.pi*sigma))*np.exp(-(x-mu)**2/(2*sigma))

fig1=plt.figure(figsize=[20,10],dpi=300)
plt.plot(x[:,0],p[:,0],c="r")
plt.plot(x[:,0],p[:,1],c="g")
plt.plot(x[:,0],p[:,2],c="b")
plt.ylabel('Histogram',fontsize=20)
plt.xlabel('x',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

fig1=plt.figure(figsize=[20,10],dpi=300)
plt.bar(x[:,0],np.sum(p,axis=1),width=50/Nb,facecolor='r',edgecolor=None,linewidth=0.5)
plt.ylabel('Histogram',fontsize=20)
plt.xlabel('x',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
# fig1.savefig('P3_6.svg',format='svg')
