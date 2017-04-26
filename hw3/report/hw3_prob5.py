import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

pic = []
with open("activate1.txt", "r") as fin:
	for line in fin:
		pic.append( list( map( float, line.rstrip().split() ) ) )
pic = np.array(pic).reshape(-1, 48, 48)
gs1 = gridspec.GridSpec(8, 8)
fig = plt.figure()
plt.clf()
for i in range(pic.shape[0]):
	ax = fig.add_subplot(gs1[i])
	ax.set_aspect(1)
	res = ax.imshow(pic[i, 4:-4, 4:-4], cmap='Reds')
	plt.axis('off')
plt.savefig('activate.png', format='png')

