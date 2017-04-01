#!/usr/bin/python
import sys
import numpy

# random seed
numpy.random.seed(3318)

# handle training data I/O
data = []
with open(sys.argv[1], "r") as fin, open(sys.argv[2], "r") as finY:
    for lineNum, line in enumerate(fin):
        if lineNum == 0:
            continue
        lineData = list( map( int, line.rstrip().split(",") ) )
        lineData.append(1)
        data.append(lineData)
    for lineNum, line in enumerate(finY):
        data[lineNum].append( int(line) )
data = numpy.array(data, dtype = numpy.int32)
print("\x1b[0;31;40m<I/O> Training Data Ready\x1b[0m")
print("Shape |", data.shape)

# tabel info
# x is "age", y is "class"
# maximize p(x1,y1)p(x2, y2)... = p(y1|x1)p(x1)p(y2|x2)p(x2)...
# target is p(y|x)
table = numpy.zeros( (data[:, 0].max(axis = 0) + 1, 2) )
for i in range(data.shape[0]):
    table[ data[i, 0], data[i, -1] ] += 1.
correct = 0
for i in range(data.shape[0]):
    if data[i, 0] >= table.shape[0]:
        correct += numpy.random.randint(2)
    elif table[data[i, 0], 0] > table[data[i, 0], 1] and data[i, -1] == 0:
        correct += 1
    elif table[data[i, 0], 0] <= table[data[i, 0], 1] and data[i, -1] == 1:
        correct += 1
print("\x1b[0;32;40m<Optimization Finished> Finish Training - Accuracy:" + str(correct/data.shape[0]) + "\x1b[0m")

# handle testing data I/O
dataTest = []
with open(sys.argv[3], "r") as fin:
    for lineNum, line in enumerate(fin):
        if lineNum == 0:
            continue
        lineData = list( map( int, line.rstrip().split(",") ) )
        lineData.append(1)
        dataTest.append(lineData)
dataTest = numpy.array(dataTest, dtype = numpy.int32)
print("\x1b[0;31;40m<I/O> Testing Data Ready\x1b[0m")
print("Shape |", dataTest.shape)

# output
with open(sys.argv[4], "w") as fout:
    fout.write("id,label\n")
    for i in range(dataTest.shape[0]):
        if dataTest[i, 0] >= table.shape[0]:
            fout.write(str(i + 1) + "," + str( numpy.random.randint(2) ) + "\n")
        elif table[dataTest[i, 0], 0] > table[dataTest[i, 0], 1]:
            fout.write(str(i + 1) + ",0\n")
        else:
            fout.write(str(i + 1) + ",1\n")
print("\x1b[0;32;40m<Done> Output File Available\x1b[0m")

