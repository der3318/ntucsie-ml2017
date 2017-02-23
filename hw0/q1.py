#!/usr/bin/python
import sys
import numpy

mA = []
mB = []
with open( sys.argv[1] ) as f: # matrixA.txt
    for line in f:
        mA.append( list( map( int, line.split(",") ) ) )
with open( sys.argv[2] ) as f: # matrixB.txt
    for line in f:
        mB.append( list( map( int, line.split(",") ) ) )
ans = numpy.dot( numpy.array(mA), numpy.array(mB) ).reshape(-1).tolist()
for element in sorted(ans):
    print(element)

