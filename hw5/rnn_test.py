from __future__ import print_function
import numpy as np
np.random.seed(3318)  # for reproducibility
import tensorflow as tf
tf.set_random_seed(3318)
import sys
import re
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.models import model_from_json
import keras.backend as K

# config
testPath = sys.argv[1]
outputPath = sys.argv[2]
tagPath = "model_rnn/tags.npy"
idxPath = "model_rnn/word2Idx.npy"
modelDir = "model_keras_rnn"
numWords = 3000
seqLen = 50

# read tags and testing data
print("\x1b[0;31;40m<I/O> Reading Testing Data\x1b[0m")
testData = []
tags = np.load(tagPath)
tag2Idx = {tags[i]: i for i in range(tags.shape[0])}
numTags = tags.shape[0]
word2Idx = np.load(idxPath).item()
with open(testPath, "r", encoding = "cp1252", errors = "replace") as fin:
    for lineNum, line in enumerate(fin):
        if lineNum > 0:
            print("\rProcessing - Line " + str(lineNum + 1) + "/1235" , end = "")
            splitedLine = line.rstrip().split(",")
            lineData = []
            for word in re.sub( "[^a-zA-Z ]+", " ", " ".join(splitedLine[1:]) ).lower().split():
                if word in word2Idx and word2Idx[word] < numWords:    lineData.append( word2Idx[word] )
            testData.append(lineData)
testData = sequence.pad_sequences(testData, maxlen = seqLen)
print("\ntags.shape tags[0] |", tags.shape, tags[0])
print("len(tag2Idx) tag2Idx[\"HUMOUR\"] |", len(tag2Idx), tag2Idx["HUMOUR"])
print("testData.shape testData[0] |", testData.shape, testData[0])

json_file = open(modelDir + "/model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(modelDir + "/model.h5")
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
predictions = model.predict(testData, verbose = 1)
print("\nprediction[0, :] |", predictions[0, :])

# output
print("\x1b[0;31;40m<Output> Writing Result to CSV File\x1b[0m")
with open(outputPath, "w") as fout:
    fout.write("\"id\",\"tags\"\n")
    for i in range(predictions.shape[0]):
        curPredictions = predictions[i, :]
        fout.write("\"" + str(i) + "\",\"")
        first = True
        for j in range(numTags):
            if curPredictions[j] < 0.5: continue
            if first:
                fout.write(tags[j])
                first = False
            else:
                fout.write(" " + tags[j])
        if first:
            fout.write(tags[np.argmax(curPredictions)])
        fout.write("\"\n")

