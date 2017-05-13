import matplotlib.pyplot as plt
import numpy as np

meanAndVar = np.load("mean_and_var2.npy")

plt.plot(np.arange(1, 61, 1), meanAndVar[:,1], 'ro')
plt.xlabel('Input Dimension (d_i)')
plt.ylabel('var(d_i)')
plt.show()
plt.savefig("dim.png", format = "png")

