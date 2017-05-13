import sys
import numpy as np

# config
filename = sys.argv[1]
meanAndVar = np.load("mean_and_var2.npy")
output = sys.argv[2]

# mean and var
print("meanAndVar.shape |", meanAndVar.shape)
print("meanAndVar[0, :] |", meanAndVar[0, :])

# read data
data = np.load(filename)
print("data[\"0\"].shape |", data["0"].shape)

# deal with 200 datasets
logDim = []
for i in range(200):
    x = data[str(i)]
    var = np.var(x)
    dist = (meanAndVar[:, 1] - var) ** 2
    dim = np.argmin(dist) + 1
    logDim.append( np.log(dim) )
    print("\rNo." + str(i + 1) + "/200 - dim = " + str(dim), end = "")
print("")

# output
with open(output, "w") as fout:
    fout.write("SetId,LogDim\n")
    for idx, ans in enumerate(logDim):
        fout.write(str(idx) + "," + str(ans) + "\n")

