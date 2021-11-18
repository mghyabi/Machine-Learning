import numpy as np
import matplotlib.pyplot as plt
import os

#a)
M=9

x=np.arange(1e4+1)/1e4
t=np.sin(x*2*np.pi)

ad=os.getcwd()
ad=ad+'\data\\'
Train=np.loadtxt(ad+'curvefitting.txt', unpack='True')
Train=Train.T

Test=np.tile(np.arange(1e4+1)/1e4,(2,1)).T
Test[:,1]=0

for Lambda in np.exp([0,-18]):
    phi=np.zeros((np.ma.size(Train, axis=0),M+1))
    phi_test=np.zeros((np.ma.size(Test, axis=0),M+1))
    for i in np.arange(M+1):
        phi[:,i]=Train[:,0]**i
    
    Hphi=np.matrix(phi).getH()
    W=(np.linalg.inv(Lambda*np.eye(M+1)+Hphi@phi)@Hphi)@Train[:,1]
    
    for i in np.arange(M+1):
        phi_test[:,i]=Test[:,0]**i

    Test[:,1,None]=phi_test@W.T
    
    
    
    fig=plt.figure(figsize=[10,7])
    plt.plot(x,t,c="g")
    plt.plot(Test[:,0],Test[:,1],c='r')
    plt.scatter(Train[:,0], Train[:,1], alpha=0.8,s=100, facecolor='none', edgecolors='blue', linewidth=2, marker="o")
    plt.xlabel('x',fontsize=20)
    plt.ylabel('t',fontsize=20)
    plt.text(0.7,1,'ln $\lambda$ = ' + str(np.log(Lambda)) ,fontsize=18)
    plt.ylim([-1.5,1.5])
    plt.xticks([0,1],fontsize=15)
    plt.yticks([-1,0,1],fontsize=15)
    plt.show()
#     fig.savefig('p5_a_l'+str(np.log(Lambda))+'.svg',format='svg')
#     fig.savefig('p5_a_l'+str(np.log(Lambda))+'.tif')
