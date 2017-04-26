import numpy as np
import matplotlib.pyplot as plt

conf_arr = [[ 2987,    58,   955,   832,  1991,   181,   986],
		[  142,   218,   195,    44,   235,    11,    27],
		[  524,     8,  3482,   478,  2406,   708,   588],
		[  140,     0,   386, 12491,   793,   182,   438],
		[  368,    15,   883,   635,  6331,    71,  1357],
		[   77,     0,   942,   394,   223,  4530,   176],
		[  224,     7,   601,   929,  2309,    99,  5761]]
conf_arr = np.transpose( np.array(conf_arr) )

norm_conf = []
for i in conf_arr:
    a = 0
    tmp_arr = []
    a = sum(i, 0)
    for j in i:
        tmp_arr.append(float(j)/float(a))
    norm_conf.append(tmp_arr)

fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                interpolation='nearest')

width, height = conf_arr.shape

for x in range(width):
    for y in range(height):
        ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center')

cb = fig.colorbar(res)
alphabet = '0123456789'
plt.xticks(range(width), alphabet[:width])
plt.yticks(range(height), alphabet[:height])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.png', format='png')
