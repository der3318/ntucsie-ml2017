#!/usr/bin/python
import sys
import numpy

numpy.random.seed(3318)
# handle training data I/O
data = []
h24 = []
lineNum = 0
with open(sys.argv[1], "r", encoding = "cp1252") as fin:
    for lineNum, line in enumerate(fin):
        if lineNum == 0:
            continue
        lineData = list( map(float, line.rstrip().replace("NR", "0.01").split(",")[3:]) )
        if (lineNum % 18) == 10:
            for i in range(24):
                if lineData[i] < 0:
                    data.append( data[-1] )
                else:
                    data.append([ lineData[i] ])
data = numpy.array(data)
dataNormed = data / data.max(axis = 0)
maxPM = data.max(axis = 0)[-1]
print("Training Data Size:", data.shape)

# handle testing data
dataTest = []
with open(sys.argv[2], "r", encoding = "cp1252") as fin:
    for lineNum, line in enumerate(fin):
        lineData = list( map(float, line.rstrip().replace("NR", "0.01").split(",")[2:]) )
        if (lineNum % 18) == 9:
            for i in range(9):
                if lineData[i] < 0:
                    dataTest.append( dataTest[-1] )
                else:
                    dataTest.append([ lineData[i] ])
dataTest = numpy.array(dataTest)
dataTestNormed = dataTest / data.max(axis = 0)
print("Testing Data Size:", dataTest.shape)

# initial configurations
dim = 1 * 8
lr = 0.8
b = 0.1
lrB = 0.0
w = []
lrW = numpy.zeros(dim)
w = numpy.array([(1 + i // 4) for i in range(dim)])

# sgd process
maxIt = 2000000
batchSize = 10
curIdx = 0
curTestIdx = 0
rndPerm = numpy.random.permutation(data.shape[0] - 8)
rndTestPerm = numpy.random.permutation(dataTest.shape[0] // 9)
for i in range(maxIt):
    if (i % 10000) == 9999:
        lost = 0.0
        for j in range(data.shape[0] - 8):
            trainX = dataNormed[j:j + 8,:].reshape(-1)
            trainY = data[j + 8, -1]
            lost += ( trainY - maxPM * ( b + numpy.dot(trainX, w) ) ) ** 2
        for j in list( range(0, dataTest.shape[0], 9) ):
            trainX = dataTestNormed[j:j + 8,:].reshape(-1)
            trainY = dataTest[j + 8, -1]
            lost += ( trainY - maxPM * ( b + numpy.dot(trainX, w) ) ) ** 2
        lost = lost / (data.shape[0] - 8 + dataTest.shape[0] // 9)
        print("\rIteration No." + str(i + 1) + "/" + str(maxIt) + " => LOST = " + str(lost), end = "")
    graB = 0.0
    graW = 0.0
    if i % 4 == 0:
        for bat in range(batchSize):
            curRnd = rndTestPerm[curTestIdx + bat]
            x = dataTestNormed[curRnd * 9:curRnd * 9 + 8,:].reshape(-1)
            y = dataTestNormed[curRnd * 9 + 8, -1]
            total = b + numpy.dot(x, w)
            graB = graB - 2.0 * (y - total)
            graW = graW - 2.0 * (y - total) * x
        lrB = lrB + graB ** 2
        lrW = lrW + graW ** 2
        b = b - lr / numpy.sqrt(lrB) * graB
        w = w - lr / numpy.sqrt(lrW) * graW
        if curTestIdx + 2 * batchSize >= rndTestPerm.size:
            curTestIdx = 0
        else:
            curTestIdx = curTestIdx + batchSize
    else:
        for bat in range(batchSize):
            curRnd = rndPerm[curIdx + bat]
            x = dataNormed[curRnd:curRnd + 8,:].reshape(-1)
            y = dataNormed[curRnd + 8, -1]
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

# output result
with open(sys.argv[3], "w") as fout:
    fout.write("id,value\n")
    for i in range(dataTest.shape[0] // 9):
        x = dataTestNormed[9 * i + 1:9 * i + 9,:].reshape(-1)
        y = maxPM * ( b + numpy.dot(x, w) )
        fout.write("id_" + str(i) + "," + str( int(y) ) + "\n")

