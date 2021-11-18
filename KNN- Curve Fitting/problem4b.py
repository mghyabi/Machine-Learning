import numpy as np
import matplotlib.pyplot as plt
import os

#b)
Lambda=0

x=np.arange(1e4+1)/1e4
t=np.sin(x*2*np.pi)[np.newaxis]

ad=os.getcwd()
ad=ad+'\data\\'
Train=np.loadtxt(ad+'curvefitting.txt', unpack='True')
Train=Train.T

Test=np.tile(np.arange(1e4+1)/1e4,(2,1)).T
Test[:,1]=0

Ms=np.arange(10)
E_test=np.zeros((1,np.size(Ms)))
E_train=np.zeros((1,np.size(Ms)))
for M in Ms:
    phi=np.zeros((np.ma.size(Train, axis=0),M+1))
    phi_test=np.zeros((np.ma.size(Test, axis=0),M+1))
    for i in np.arange(M+1):
        phi[:,i]=Train[:,0]**i
    
    Hphi=np.matrix(phi).getH()
    W=(np.linalg.inv(Lambda*np.eye(M+1)+Hphi@phi)@Hphi)@Train[:,1]
    
    for i in np.arange(M+1):
        phi_test[:,i]=Test[:,0]**i

    Test[:,1,None]=phi_test@W.T
    E_train[0,M]=np.sqrt(0.5*np.sum(np.multiply(Train[:,1,None]-phi@W.T,Train[:,1,None]-phi@W.T))/np.ma.size(Train, axis=0))
    E_test[0,M]=np.sqrt(0.5*np.sum((Test[:,1,None]-t.T)**2)/Test.shape[0])
    
fig=plt.figure(figsize=[10,7])
plt.plot(Ms,E_test[0,:],c="r",marker="o",markersize=10,fillstyle='none',label='Test')
plt.plot(Ms,E_train[0,:],c="b",marker="o",markersize=10,fillstyle='none',label='Training')
plt.xlabel('M',fontsize=20)
plt.ylabel('$\mathregular{E_{RMS}}$',fontsize=20)
plt.ylim([0,1.0])
plt.xlim([-1,10])
plt.xticks(np.arange(0,10,3),fontsize=15)
plt.yticks([0,0.5,1.0],fontsize=15)
plt.legend(fontsize=15,loc='upper left')
plt.show()
# fig.savefig('p4_b.svg',format='svg')
# fig.savefig('p4_b.tif')
