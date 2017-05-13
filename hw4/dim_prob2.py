import sys
import os
import re
from scipy import misc
import numpy as np

# config
path = "hand"
meanAndVar = np.load("mean_and_var2.npy")
rows = 480
cols = 512

# img collection
imgs = []
for png in os.listdir(path):
    if not re.match(".*\.png", png): continue
    imgs.append( misc.imread(os.path.join(path, png), flatten = 0) )
imgs = np.array(imgs, np.float32).reshape( (-1, rows * cols) )
print("imgs.shape |", imgs.shape)

# mean and var
print("meanAndVar.shape |", meanAndVar.shape)
print("meanAndVar[0, :] |", meanAndVar[0, :])

var = np.var(imgs)
dist = (meanAndVar[:, 1] - var) ** 2
dim = np.argmin(dist) + 1
print( "var = " + str(var) + ", dim = " + str(dim) + ", var(dim) = " + str(meanAndVar[dim - 1, 1]) )

