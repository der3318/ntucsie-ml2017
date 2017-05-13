import os
import re
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# config
path = "data_pca"
output = "output_pca"
rows = 64
cols = 64
k1 = 9
k2 = 5

# img collection
imgs = []
for bmp in os.listdir(path):
    if not re.match("[A-J]0[0-9]\.bmp", bmp): continue
    imgs.append( misc.imread(os.path.join(path, bmp), flatten = 0) )
imgs = np.array(imgs, np.float32).reshape( (-1, rows * cols) )
print("imgs.shape |", imgs.shape)

# svd
mean = np.mean(imgs, axis = 0)
U, s, Vt = np.linalg.svd(imgs - mean, full_matrices = False)
V = Vt.T
S = np.diag(s)

# k1 = 9
gs = gridspec.GridSpec(k1 // 3, 3)
fig = plt.figure()
plt.clf()
imgs1 = np.dot( U[:, :k1], np.dot(S[:k1, :k1], V[:, :k1].T) ) + mean
print("k1 = 9, RMSE =", np.mean( (imgs - imgs1) ** 2 ) ** .5)
for i in range(k1):
    ax = fig.add_subplot(gs[i])
    ax.set_aspect(1)
    res = ax.imshow(-V[:, i].reshape(rows, cols), cmap = "Greys")
    plt.axis("off")
plt.savefig(os.path.join(output, "pca1_eig.png"), format = "png")
fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(-mean.reshape(rows, cols), cmap = "Greys")
plt.axis("off")
plt.savefig(os.path.join(output, "pca1_avg.png"), format = "png")

# k2 = 5
gs = gridspec.GridSpec(imgs.shape[0] // 10, 10)
fig = plt.figure()
plt.clf()
for i in range(imgs.shape[0]):
    ax = fig.add_subplot(gs[i])
    ax.set_aspect(1)
    res = ax.imshow(-imgs[i, :].reshape(rows, cols), cmap = "Greys")
    plt.axis("off")
plt.savefig(os.path.join(output, "pca2_ori.png"), format = "png")
fig = plt.figure()
plt.clf()
imgs2 = np.dot( U[:, :k2], np.dot(S[:k2, :k2], V[:, :k2].T) ) + mean
print("k2 = 5, RMSE =", np.mean( (imgs - imgs2) ** 2 ) ** .5)
for i in range(imgs.shape[0]):
    ax = fig.add_subplot(gs[i])
    ax.set_aspect(1)
    res = ax.imshow(-imgs2[i, :].reshape(rows, cols), cmap = "Greys")
    plt.axis("off")
plt.savefig(os.path.join(output, "pca2_rec.png"), format = "png")

# error < 0.01 * 256
for k in range(10, 1000):
    imgsk = np.dot( U[:, :k], np.dot(S[:k, :k], V[:, :k].T) ) + mean
    rmse = np.mean( (imgs - imgsk) ** 2 ) ** .5
    print("\rk = " + str(k) + ", RMSE = " + str(rmse), end = "")
    if rmse < 0.01 * 256: break
print("")
gs = gridspec.GridSpec(imgs.shape[0] // 10, 10)
fig = plt.figure()
plt.clf()
for i in range(imgs.shape[0]):
    ax = fig.add_subplot(gs[i])
    ax.set_aspect(1)
    res = ax.imshow(-imgsk[i, :].reshape(rows, cols), cmap = "Greys")
    plt.axis("off")
plt.savefig(os.path.join(output, "pca3.png"), format = "png")

