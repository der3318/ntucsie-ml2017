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
        lineData = list( map( float, line.rstrip().split(",") ) )
        lineData.append(1.)
        data.append(lineData)
    for lineNum, line in enumerate(finY):
        data[lineNum].append(2. * float(line) - 1.)
originData = numpy.array(data)
data = originData / originData.max(axis = 0)
print("\x1b[0;31;40m<I/O> Training Data Ready\x1b[0m")
print("Shape |", data.shape)

# initial configurations
dim = data.shape[1] - 1
lr = 0.0001
w = numpy.array([0.1 for i in range(dim)])

# sgd process
print("\x1b[0;31;40m<Training> Logistic Regression\x1b[0m")
progress1 = ["[| \x1b[0;32;40m>\x1b[0m", "[| \x1b[0;32;40m>>\x1b[0m", "[| \x1b[0;32;40m>>>\x1b[0m", "[| \x1b[0;32;40m>>>>\x1b[0m", "[| \x1b[0;32;40m>>>>>\x1b[0m", "[| \x1b[0;32;40m>>>>>>\x1b[0m", "[| \x1b[0;32;40m>>>>>>>\x1b[0m", "[| \x1b[0;32;40m>>>>>>>>\x1b[0m", "[| \x1b[0;32;40m>>>>>>>>>\x1b[0m", "[| \x1b[0;32;40m>>>>>>>>>>\x1b[0m"]
progress2 = ["\x1b[0;31;40m---------\x1b[0m |]", "\x1b[0;31;40m--------\x1b[0m |]", "\x1b[0;31;40m-------\x1b[0m |]", "\x1b[0;31;40m------\x1b[0m |]", "\x1b[0;31;40m-----\x1b[0m |]", "\x1b[0;31;40m----\x1b[0m |]", "\x1b[0;31;40m---\x1b[0m |]", "\x1b[0;31;40m--\x1b[0m |]", "\x1b[0;31;40m-\x1b[0m |]", " |]"]
epoch = 1500
batchSize = 10000
curBatch = 0
x = numpy.zeros( (batchSize, dim) )
y = numpy.zeros( (batchSize, 1) )
for step in range(epoch):
    rndPerm = numpy.random.permutation(data.shape[0])
    for curIdx, rnd in enumerate(rndPerm):
        x[curBatch, :] = data[rnd, :-1]
        y[curBatch, 0] = data[rnd, -1]
        curBatch += 1
        if curBatch >= batchSize:
            w = w + lr * numpy.sum(y * x  / (numpy.exp( y * numpy.dot( x, numpy.reshape( w, (-1, 1) ) ) ) + 1), axis = 0)
            curBatch = 0
    pro = step * 10 // epoch # calculate the progressing percentage
    pred = 1 / (numpy.exp( -1 * numpy.dot( data[:, :-1], numpy.reshape( w, (-1, 1) ) ) ) + 1) - 0.5
    correct = 0
    for i in range(data.shape[0]):
        correct = (correct + 1) if pred[i, 0] * data[i, -1] >= 0 else correct
    print("\r0%" + progress1[pro] + progress2[pro] + "100% - Epoch: " + str(step + 1) + "/" + str(epoch) + " - Accuracy: " + str(correct / data.shape[0]), end = "")
print("\n\x1b[0;32;40m<Optimization Finished> Finish Training\x1b[0m")

# handle testing data I/O
dataTest = []
with open(sys.argv[3], "r") as fin:
    for lineNum, line in enumerate(fin):
        if lineNum == 0:
            continue
        lineData = list( map( float, line.rstrip().split(",") ) )
        lineData.append(1.)
        dataTest.append(lineData)
dataTest = numpy.array(dataTest)
dataTest = dataTest / originData[:, :-1].max(axis = 0)
print("\x1b[0;31;40m<I/O> Testing Data Ready\x1b[0m")
print("Shape |", dataTest.shape)

# output
pred = 1 / (numpy.exp( -1 * numpy.dot( dataTest, numpy.reshape( w, (-1, 1) ) ) ) + 1) - 0.5
with open(sys.argv[4], "w") as fout:
    fout.write("id,label\n")
    for i in range(pred.shape[0]):
        label = "1" if pred[i, 0] >= 0 else "0"
        fout.write(str(i + 1) + "," + label + "\n")
print("\x1b[0;32;40m<Done> Output File Available\x1b[0m")

