#P1
for name in dir():
    del globals()[name]
    
import numpy as np
import matplotlib.pyplot as plt
import os

#import from data folder
ad=os.getcwd()
ad=ad+'\data\\'
X=np.loadtxt(open(ad+"out.csv", "rb"), delimiter=",")

labels=np.array(['Dove', 'Hen', 'Duck', 'Goose', 'Owl', 'Hawk', 'Eagle',
                'Fox', 'Dog', 'Wolf', 'Cat', 'Tiger', 'Lion', 'Horse',
                'Zebra', 'Cow', 'Bear'])
Traits=np.array(['small', 'medium', 'large', '2 legs', '4 legs', 'hair',
                 'hooves','mane', 'feathers', 'hunt', 'run', 'fly', 'swim'])

MapDim=int(np.sqrt(X.shape[0]))

DMap=np.reshape(X[:,0],(MapDim,MapDim),order='F')
DMap=-DMap
DMap=DMap-np.min(DMap)
DMap=DMap/np.max(DMap)
DMap=20**DMap
DMap=DMap-np.min(DMap)
DMap=DMap/np.max(DMap)

X=X[:,1:]
k=np.argmax(X,axis=0)

#making the grid
nodes=np.linspace(0,MapDim-1,MapDim)
x1, y1 = np.meshgrid(nodes, nodes)
i=np.arange(1,MapDim,2)
x1[i,:]=x1[i,:]+0.5

#plotting
fig1=plt.figure(figsize=[10,10],dpi=300)
plt.scatter(x1, y1, alpha=0.8,marker='h', facecolor='none', edgecolors='k', s=1700*DMap)
for j in np.arange(len(labels)):
    c=np.array([k[j]//MapDim,k[j]%MapDim]) # getting the winner
    c=c*2/2
    if c[1]%2==1:
        c[0]=c[0]+ 0.5
    plt.text(c[0]-0.35,c[1]+0.1,labels[j], fontsize=15)
plt.gca().invert_yaxis()
plt.axis('off')
plt.show(fig1)
fig1.savefig('P1_1.svg',format='svg')

#_______
X=X[:,17:]
k=np.argmax(X,axis=0)

Map1=np.reshape(X[:,0],(MapDim,MapDim),order='F')-np.reshape(X[:,1],(MapDim,MapDim),order='F')
Map2=np.reshape(X[:,0],(MapDim,MapDim),order='F')-np.reshape(X[:,2],(MapDim,MapDim),order='F')
Map3=np.reshape(X[:,1],(MapDim,MapDim),order='F')-np.reshape(X[:,2],(MapDim,MapDim),order='F')

fig2=plt.figure(figsize=[10,10],dpi=300)
plt.scatter(x1, y1, alpha=0.8,marker='h', facecolor='none', edgecolors='k', s=1700*DMap)
plt.scatter(x1[(Map1>=0) * (Map2>=0)], y1[(Map1>=0) * (Map2>=0)], alpha=0.8,marker='h', c='r', s=200)
plt.scatter(x1[(Map2<=0) * (Map3<=0)], y1[(Map2<=0) * (Map3<=0)], alpha=0.8,marker='h', c='g', s=200)
plt.scatter(x1[(Map1<0) * (Map3>0)], y1[(Map1<0) * (Map3>0)], alpha=0.8,marker='h', c='b', s=200)
for j in np.arange(3):
    c=np.array([k[j]//MapDim,k[j]%MapDim])
    c=c*2/2
    if c[1]%2==1:
        c[0]=c[0]+ 0.5
    plt.text(c[0]-0.35,c[1]+0.1,Traits[j], fontsize=15)
plt.gca().invert_yaxis()
plt.axis('off')
plt.show(fig2)
# fig2.savefig('P1_2.svg',format='svg')

#_______

Map1=np.reshape(X[:,10],(MapDim,MapDim),order='F')-np.reshape(X[:,11],(MapDim,MapDim),order='F')
Map2=np.reshape(X[:,10],(MapDim,MapDim),order='F')-np.reshape(X[:,12],(MapDim,MapDim),order='F')
Map3=np.reshape(X[:,11],(MapDim,MapDim),order='F')-np.reshape(X[:,12],(MapDim,MapDim),order='F')

fig3=plt.figure(figsize=[10,10],dpi=300)
plt.scatter(x1, y1, alpha=0.8,marker='h', facecolor='none', edgecolors='k', s=1700*DMap)
plt.scatter(x1[(Map1>=0) * (Map2>=0)], y1[(Map1>=0) * (Map2>=0)], alpha=0.8,marker='h', c='r', s=200)
plt.scatter(x1[(Map2<=0) * (Map3<=0)], y1[(Map2<=0) * (Map3<=0)], alpha=0.8,marker='h', c='g', s=200)
plt.scatter(x1[(Map1<0) * (Map3>0)], y1[(Map1<0) * (Map3>0)], alpha=0.8,marker='h', c='b', s=200)
for j in np.array([10,11,12]):
    c=np.array([k[j]//MapDim,k[j]%MapDim])
    c=c*2/2
    if c[1]%2==1:
        c[0]=c[0]+ 0.5
    plt.text(c[0]-0.35,c[1]+0.1,Traits[j], fontsize=15)
plt.gca().invert_yaxis()
plt.axis('off')
plt.show(fig3)
# fig3.savefig('P1_3.svg',format='svg')

#_______

Map1=np.reshape(X[:,5],(MapDim,MapDim),order='F')-np.reshape(X[:,8],(MapDim,MapDim),order='F')

fig4=plt.figure(figsize=[10,10],dpi=300)
plt.scatter(x1, y1, alpha=0.8,marker='h', facecolor='none', edgecolors='k', s=1700*DMap)
plt.scatter(x1[Map1>=0], y1[Map1>=0], alpha=0.8,marker='h', c='r', s=200)
plt.scatter(x1[Map1<0], y1[Map1<0], alpha=0.8,marker='h', c='g', s=200)
for j in np.array([5,8]):
    c=np.array([k[j]//MapDim,k[j]%MapDim])
    c=c*2/2
    if c[1]%2==1:
        c[0]=c[0]+ 0.5
    plt.text(c[0]-0.35,c[1]+0.1,Traits[j], fontsize=15)
plt.gca().invert_yaxis()
plt.axis('off')
plt.show(fig4)
# fig4.savefig('P1_4.svg',format='svg')
