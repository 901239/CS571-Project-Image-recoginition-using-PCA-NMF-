# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 00:16:20 2021

@author: prajy
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import decomposition
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from PIL import Image
from scipy.ndimage import  rotate


P=np.zeros((4000,4,4))
P[0:199]=np.array([[0,1,1,0],[1,0,0,1],[1,1,1,1],[1,0,0,1]])   ##A
P[200:399]=np.array([[0,1,1,1],[1,0,0,0],[1,0,0,0],[0,1,1,1]]) ##C
P[400:599]=np.array([[1,1,1,0],[1,0,0,1],[1,0,0,1],[1,1,1,0]]) ##D
P[600:799]=np.array([[1,1,1,1],[1,0,0,0],[1,1,1,1],[1,0,0,0]]) ##F
P[800:999]=np.array([[0,1,1,0],[1,0,0,0],[1,0,1,1],[0,1,1,0]]) ##G
P[1000:1199]=np.array([[1,0,0,1],[1,0,0,1],[1,1,1,1],[1,0,0,1]]) ##H
P[1200:1399]=np.array([[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0]]) ##I
P[1400:1599]=np.array([[0,0,0,1],[0,0,0,1],[1,0,0,1],[1,1,1,1]]) ##J
P[1600:1799]=np.array([[1,0,0,1],[1,0,1,0],[1,1,1,0],[1,0,0,1]]) ##K
P[1800:1999]=np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,1,1,1]]) ##L
P[2000:2199]=np.array([[1,0,0,1],[1,1,0,1],[1,0,1,1],[1,0,0,1]]) ##N
P[2200:2399]=np.array([[1,1,1,1],[1,0,0,1],[1,0,0,1],[1,1,1,1]]) ##O
P[2400:2599]=np.array([[1,1,1,1],[1,0,0,1],[1,1,1,1],[1,0,0,0]]) ##P
P[2600:2799]=np.array([[1,1,1,0],[1,0,1,0],[1,1,1,0],[0,0,1,1]]) ##Q
P[2800:2999]=np.array([[1,1,1,1],[1,0,0,1],[1,1,1,0],[1,0,0,1]]) ##R
P[3000:3199]=np.array([[1,1,1,0],[0,1,0,0],[0,1,0,0],[0,1,0,0]]) ##T
P[3200:3399]=np.array([[0,1,1,0],[0,1,0,0],[0,0,1,0],[0,1,1,0]]) ##S
P[3400:3599]=np.array([[1,0,0,1],[1,0,0,1],[1,0,0,1],[0,1,1,0]]) ##U
P[3600:3799]=np.array([[0,1,0,1],[0,1,0,1],[0,1,0,1],[0,0,1,0]]) ##V
P[3800:3999]=np.array([[1,0,0,1],[0,1,1,0],[0,1,1,0],[1,0,0,1]]) ##X
#P[4000:4199]=np.array([[0,1,0,1],[0,1,0,1],[0,0,1,0],[0,0,1,0]]) ##Y
#P[4200:4399]=np.array([[1,1,1,0],[0,0,1,0],[0,1,0,0],[0,1,1,1]]) ##Z



P= shuffle(P[:4000])
fig, ax = plt.subplots(5,5, figsize = (10,10))
plt.title('Original Images')
plt.tight_layout()
axes = ax.flatten()
for i in range(25):
    axes[i].imshow(P[i],cmap="Greys")


Siz=P[0:199] #size track

N_Z=np.zeros((4000,4,4))#For storing noissy image

#Noise_1
Mean=1
varience=0.1
noise1=np.random.normal(Mean,varience,Siz.shape)
P[0:199]=P[0:199]+noise1
P[800:999]=P[800:999]+noise1
P[600:799]=P[600:799]+noise1
P[400:599]=P[400:599]+noise1
P[200:399]=P[200:399]+noise1

#Noise_2
Mean=1
varience=0.2
noise2=np.random.normal(Mean,varience,Siz.shape)
P[1400:1599]=P[1400:1599]+noise2
P[1800:1999]=P[1800:1999]+noise2
P[1200:1399]=P[1200:1399]+noise2
P[1600:1799]=P[1600:1799]+noise2
P[1200:1399]=P[1200:1399]+noise2

#Noise_3
Mean=1
varience=0.08
noise3=np.random.normal(Mean,varience,Siz.shape)
P[2600:2799]=P[2600:2799]+noise3
P[2800:2999]=P[2800:2999]+noise3
P[2400:2599]=P[2400:2599]+noise3
P[2000:2199]=P[2000:2199]+noise3
P[2200:2399]=P[2200:2399]+noise3

#Noise_4
Mean=1
varience=0.3
noise4=np.random.normal(Mean,varience,Siz.shape)
P[3800:3999]=P[3800:3999]+noise4
P[3600:3799]=P[3600:3799]+noise4
P[3400:3599]=P[3400:3599]+noise4
P[3200:3399]=P[3200:3399]+noise4
P[3000:3199]=P[3000:3199]+noise4
#Noise_5
Mean=0;
varience=0.06
#noise5=np.random.normal(Mean,var,Siz.shape)
#P[800:999]=P[800:999]+noise5
#P[800:999]=P[800:999]+noise5
#P[800:999]=P[800:999]+noise5
#P[800:999]=P[800:999]+noise5
#P[800:999]=P[800:999]+noise5
#P[800:999]=P[800:999]+noise5


N_P=P

fig, ax = plt.subplots(5,5, figsize = (10,10))
plt.title('Noisy Image')
fig.suptitle('Noisy Images', fontsize=16)
plt.tight_layout()
axes = ax.flatten()
for i in range(25):
    axes[i].imshow(N_P[i],cmap="Greys")

A=N_P.reshape(4000,4*4)
####******** Applying PCA ********####
princi_c=PCA(16)
Pc=princi_c.fit(A) #storing PCA
X=princi_c.components_ #storing components


Latest=X.reshape(16,4,4)
fig, ax = plt.subplots(5,5, figsize = (10,10))
plt.title('PCA OutPut')
plt.tight_layout()
axes = ax.flatten()
for i in range(16):
    axes[i].imshow(Latest[i],cmap="Greys")
    
    
####****** Applying NMF *****####    
non_neg = NMF(16) #applying NMF On 16 images
T = non_neg.fit(np.abs(A))
G=non_neg.components_

T=G.reshape(16,4,4)
fig, ax = plt.subplots(5,5, figsize = (10,10))
plt.title('NMF OutPut')
plt.tight_layout()
axes = ax.flatten()
for i in range(16):
    axes[i].imshow(T[i],cmap="Greys")