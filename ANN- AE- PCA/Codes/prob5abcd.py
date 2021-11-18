for name in dir():
    del globals()[name]

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

#import images from data folder
ad=os.getcwd()
ad=ad+'\data\\'

X=np.loadtxt(open(ad+"spikes.csv", "rb"), delimiter=",")

#____________________
#P5_b

pca = PCA(n_components=1)
PC1D = pca.fit_transform(X)
print(pca.explained_variance_ratio_)
fig=plt.figure(figsize=[10,5],dpi=300)
plt.scatter(PC1D[:,0], np.zeros((1,PC1D.shape[0])), alpha=0.8, c='blue', edgecolors='none', s=20)
plt.axis([np.min(PC1D),np.max(PC1D),-0.5,0.5])
plt.grid('True',linestyle='--', linewidth=1)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlabel('$x_1$',fontsize=20)
plt.xticks(fontsize=15)
plt.show()
# fig.savefig('p5_b.svg',format='svg')

#____________________
#P5_c

pca = PCA(n_components=2)
PC2D = pca.fit_transform(X)
print(pca.explained_variance_ratio_)
fig=plt.figure(figsize=[10,10],dpi=300)
plt.scatter(PC2D[:,0], PC2D[:,1], alpha=0.8, c='blue', edgecolors='none', s=20)
plt.axis([-1.5e-4,2e-4,-1.5e-4,2e-4])
plt.grid('True',linestyle='--', linewidth=1)
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.xlabel('$x_1$',fontsize=20)
plt.ylabel('$x_2$',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
# fig.savefig('p5_c.svg',format='svg')

#____________________
#P5_d

pca = PCA(n_components=3)
PC3D = pca.fit_transform(X)
print(pca.explained_variance_ratio_)
fig=plt.figure(figsize=[10,10],dpi=300)
ax = plt.axes(projection='3d')
ax.scatter3D(PC3D[:,0], PC3D[:,1],PC3D[:,2], alpha=0.8, c='blue', edgecolors='none', s=20)
ax.set_xlabel('X1',fontsize=15)
ax.set_ylabel('X2',fontsize=15)
ax.set_zlabel('X3',fontsize=15)
# ax.view_init(elev=20., azim=60)
plt.show()
# fig.savefig('p5_d.svg',format='svg')
