import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

pic = []
with open("saliency.txt", "r") as fin:
	for line in fin:
		pic.append( np.array( list( map( float, line.rstrip().split() ) ) ).reshape(48, 48) )
result = np.zeros([48, 48])
for row in range(48):
	for col in range(48):
		result[row, col] = pic[0][0, 0] if np.absolute(pic[1][row, col] * 10000) < 0.06 else pic[0][row, col]
gs1 = gridspec.GridSpec(2, 2)
fig = plt.figure()
plt.clf()
ax = fig.add_subplot(gs1[0])
ax.set_aspect(1)
res = ax.imshow(1 - pic[0], cmap='Greys')
plt.axis('off')
ax = fig.add_subplot(gs1[1])
ax.set_aspect(1)
res = ax.imshow(1 - result, cmap='Greys')
plt.axis('off')
ax = fig.add_subplot(gs1[2])
ax.set_aspect(1)
res = ax.imshow(np.absolute(pic[1] * 10000), cmap=plt.cm.jet, interpolation='nearest')
cb = fig.colorbar(res)
plt.axis('off')
plt.savefig('saliency.png', format='png')
