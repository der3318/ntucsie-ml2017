#!/usr/bin/python
import sys
import numpy

numpy.random.seed(3318)
# handle training data I/O
data = []
h24 = []
lineNum = 0
with open(sys.argv[1], "r", encoding = "cp1252") as fin:
    for line in fin:
        if lineNum == 0:
            lineNum += 1
            continue
        lineData = list( map(float, line.rstrip().replace("NR", "0.01").split(",")[3:]) )
        if (lineNum % 18) == 1:
            for i in range(24):
                h24.append([ lineData[i] ])
        else:
            for i in range(24):
                h24[i].append(lineData[i])
        if (lineNum % 18) == 0:
            for i in range(24):
                data.append(h24[i])
            h24 = []
        lineNum += 1
data = numpy.array(data)
data[ :, [9, -1] ] = data[ :, [-1, 9] ]
dataNormed = data / data.max(axis = 0)
maxPM = data.max(axis = 0)[-1]
print("Training Data Size:", data.shape)
print("Example Training Data (2014/1/1,0am)")
print(data[0])

# initial configurations
dim = 18 * 9
lr = 0.05
b = 0.1
lrB = 0.0
w = []
lrW = []
for i in range(dim):
    w.append( (1 + i // 72) * 0.1 )
    lrW.append(0.0)
w = numpy.array(w)
lrW = numpy.array(lrW)

# sgd process
maxIt = 500000
batchSize = 10
curIdx = 0
rndPerm = numpy.random.permutation(data.shape[0] - 9)
for i in range(maxIt):
    if (i % 10000) == 0:
        lost = 0.0
        for j in range(data.shape[0] - 9):
            trainX = dataNormed[j:j + 9,:].reshape(-1)
            trainY = data[j + 9, -1]
            lost += ( trainY - maxPM * ( b + numpy.dot(trainX, w) ) ) ** 2
        lost = lost / (data.shape[0] - 9)
        print("\rIteration No." + str(i) + " => LOST = " + str(lost), end = "")
    graB = 0.0
    graW = 0.0
    for bat in range(batchSize):
        x = dataNormed[rndPerm[curIdx + bat]:rndPerm[curIdx + bat] + 9,:].reshape(-1)
        y = dataNormed[rndPerm[curIdx + bat] + 9, -1]
        total = b + numpy.dot(x, w)
        graB = graB - 2.0 * (y - total)
        graW = graW - 2.0 * (y - total) * x
    lrB = lrB + graB ** 2
    lrW = lrW + graW ** 2
    b = b - lr / numpy.sqrt(lrB) * graB
    w = w - lr / numpy.sqrt(lrW) * graW
    if curIdx + 2 * batchSize >= rndPerm.size:
        curIdx = 0
    else:
        curIdx = curIdx + batchSize
print("\nFinal Weight and Bias")
print(w, b)

# handle testing data
dataTest = []
h9 = []
lineNum = 0
with open(sys.argv[2], "r", encoding = "cp1252") as fin:
    for line in fin:
        lineData = list( map(float, line.rstrip().replace("NR", "0.01").split(",")[2:]) )
        if (lineNum % 18) == 0:
            for i in range(9):
                h9.append([ lineData[i] ])
        else:
            for i in range(9):
                h9[i].append(lineData[i])
        if (lineNum % 18) == 17:
            for i in range(9):
                dataTest.append(h9[i])
            h9 = []
        lineNum += 1
dataTest = numpy.array(dataTest)
dataTest[ :, [9, -1] ] = dataTest[ :, [-1, 9] ]
dataTestNormed = dataTest / data.max(axis = 0)
print("Testing Data Size:", dataTest.shape)
print("Example Testing Data (id_0)")
print(dataTest[0])

# output result
with open(sys.argv[3], "w") as fout:
    fout.write("id,value\n")
    for i in range(dataTest.shape[0] // 9):
        x = dataTestNormed[9 * i:9 * i + 9,:].reshape(-1)
        y = maxPM * ( b + numpy.dot(x, w) )
        fout.write("id_" + str(i) + "," + str( int(y) ) + "\n")

