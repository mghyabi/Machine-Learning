#P4
for name in dir():
    del globals()[name]
    
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture

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
    
    E1=np.vstack((x1,y1,l1))
    E2=np.vstack((x2,y2,l2))
    E=np.hstack((E1,E2))
    return E

N=1000
d=-0.5
r=1
w=0.6
K=5

E = doublemoon(N,d,r,w).T

X_train1=E[0:500,0:2]
X_train2=E[500:,0:2]
# fit Gaussian Mixture Models
clf1 = mixture.GaussianMixture(n_components=K, covariance_type='full')
clf1.fit(X_train1)
clf2 = mixture.GaussianMixture(n_components=K, covariance_type='full')
clf2.fit(X_train2)

# display predicted scores by the model as a contour plot
x = np.linspace(-2., 3.,200)
y = np.linspace(-2., 3.,200)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z1 = -clf1.score_samples(XX)
Z1 = Z1.reshape(X.shape)
Z2 = -clf2.score_samples(XX)
Z2 = Z2.reshape(X.shape)

fig1=plt.figure(figsize=[10,10],dpi=300)
plt.contour(X, Y, Z1, levels=np.logspace(-1, .6, 15),cmap='autumn')
plt.scatter(X_train1[:,0], X_train1[:,1],s=10, alpha=0.8, c='b', edgecolors='none')
plt.grid('True',linestyle='--', linewidth=1)
plt.xlabel('$x_1$',fontsize=20)
plt.ylabel('$x_2$',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
# fig1.savefig('P4_14.svg',format='svg')

fig2=plt.figure(figsize=[10,10],dpi=300)
plt.contour(X, Y, Z2, levels=np.logspace(-1, .6, 15),cmap='autumn')
plt.scatter(X_train2[:,0], X_train2[:,1],s=10, alpha=0.8, c='b', edgecolors='none')
plt.grid('True',linestyle='--', linewidth=1)
plt.xlabel('$x_1$',fontsize=20)
plt.ylabel('$x_2$',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
# fig2.savefig('P4_15.svg',format='svg')

#____________

N=3000
S1=clf1.sample(N//2)
S1=S1[0]
S2=clf2.sample(N//2)
S2=S2[0]

fig3=plt.figure(figsize=[10,10],dpi=300)
plt.scatter(S1[:,0], S1[:,1],s=10, alpha=0.8, c='b', edgecolors='none')
plt.scatter(S2[:,0], S2[:,1],s=10, alpha=0.8, c='b', edgecolors='none')
plt.grid('True',linestyle='--', linewidth=1)
plt.axis([-2,3,-2,3])
plt.xlabel('$x_1$',fontsize=20)
plt.ylabel('$x_2$',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
# fig3.savefig('P4_16.svg',format='svg')

fig4=plt.figure(figsize=[10,10],dpi=300)
plt.scatter(S1[:,0], S1[:,1],s=10, alpha=0.8, c='b', edgecolors='none')
plt.scatter(S2[:,0], S2[:,1],s=10, alpha=0.8, c='b', edgecolors='none')
plt.scatter(E[:,0], E[:,1],s=40, alpha=0.8, c='r', edgecolors='none',marker='+')
plt.grid('True',linestyle='--', linewidth=1)
plt.axis([-2,3,-2,3])
plt.xlabel('$x_1$',fontsize=20)
plt.ylabel('$x_2$',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
# fig4.savefig('P4_8.svg',format='svg')
#_________________

N=1000

E = doublemoon(N,d,r,w).T
l=E[:,-1]
SS1=clf1.score_samples(E[:,:2])
SS2=clf2.score_samples(E[:,:2])
d=np.where(SS1>SS2,1,-1)

acc=np.sum(np.abs(d+l)/2)/E.shape[0]
print("Classifier Accuracy=%s" % acc)

fig5=plt.figure(figsize=[10,10],dpi=300)
plt.scatter(E[d+l==2,0], E[d+l==2,1], alpha=0.8,marker='+', c='b', edgecolors='none', s=75)
plt.scatter(E[d+l==-2,0], E[d+l==-2,1], alpha=0.8,marker='x', c='g', edgecolors='none', s=55)
plt.scatter(E[d+l==0,0], E[d+l==0,1], alpha=0.8,marker='*', c='r', edgecolors='none', s=95)
plt.axis([-2,3,-2,3])
plt.grid('True',linestyle='--', linewidth=1)
plt.xlabel('$x_1$',fontsize=20)
plt.ylabel('$x_2$',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show(fig5)
# fig5.savefig('P4_17.svg',format='svg')
