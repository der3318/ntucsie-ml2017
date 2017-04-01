#!/usr/bin/python
import sys
import numpy

# handle training data I/O
data = []
with open(sys.argv[1], "r") as fin, open(sys.argv[2], "r") as finY:
    for lineNum, line in enumerate(fin):
        if lineNum == 0:
            continue
        lineData = list( map( float, line.rstrip().split(",") ) )
        for i in [0, 1, 3, 4, 5]:
            lineData.append(lineData[i] ** 2)
        data.append(lineData)
    for lineNum, line in enumerate(finY):
        data[lineNum].append(2. * float(line) - 1.)
originData = numpy.array(data)
data = originData / originData.max(axis = 0)
print("\x1b[0;31;40m<I/O> Training Data Ready\x1b[0m")
print("Shape |", data.shape)

# handle testing data I/O
dataTest = []
with open(sys.argv[3], "r") as fin:
    for lineNum, line in enumerate(fin):
        if lineNum == 0:
            continue
        lineData = list( map( float, line.rstrip().split(",") ) )
        for i in [0, 1, 3, 4, 5]:
            lineData.append(lineData[i] ** 2)
        dataTest.append(lineData)
dataTest = numpy.array(dataTest)
dataTest = dataTest / originData[:, :-1].max(axis = 0)
print("\x1b[0;31;40m<I/O> Testing Data Ready\x1b[0m")
print("Shape |", dataTest.shape)

# get paramters
print("\x1b[0;31;40m<Skip> Loading Paramters\x1b[0m")
params = []
param = []
with open(sys.argv[5], "r") as fin:
    rows = 0
    isBias = False
    for line in fin:
        if rows == 0:
            rc = line.rstrip().split()
            rows = int(rc[0]) if len(rc) > 1 else 1
            isBias = False if len(rc) > 1 else True
            param = []
        else:
            rows = rows - 1
            if isBias:
                param = list( map( float, line.rstrip().split() ) )
            else:
                param.append( list( map( float, line.rstrip().split() ) ) )
            if rows == 0:
                params.append( numpy.array(param) )
print("len(params) params[14].shape params[15].shape | ", len(params), params[14].shape, params[15].shape)

# calculate
hidden1 = numpy.matmul(dataTest[:, 6:15], params[0]) + params[1]
hidden2 = numpy.matmul(dataTest[:, 15:31], params[2]) + params[3]
hidden3 = numpy.matmul(dataTest[:, 31:38], params[4]) + params[5]
hidden4 = numpy.matmul(dataTest[:, 38:53], params[6]) + params[7]
hidden5 = numpy.matmul(dataTest[:, 53:59], params[8]) + params[9]
hidden6 = numpy.matmul(dataTest[:, 59:64], params[10]) + params[11]
hidden7 = numpy.matmul(dataTest[:, 64:106], params[12]) + params[13]
hidden = numpy.concatenate( (dataTest[:, 0:6], hidden1, hidden2, hidden3, hidden4, hidden5, hidden6, hidden7, dataTest[:, 106:111]), axis = 1 )
probTest = numpy.matmul(hidden, params[14]) + params[15]

# output
with open(sys.argv[4], "w") as fout:
    fout.write("id,label\n")
    for i in range(dataTest.shape[0]):
        label = "1" if probTest[i, 1] >= probTest[i, 0] else "0"
        fout.write(str(i + 1) + "," + label + "\n")
print("\x1b[0;32;40m<Done> Output File Available\x1b[0m")

