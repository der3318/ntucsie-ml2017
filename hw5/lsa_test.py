from __future__ import print_function
import numpy as np
np.random.seed(3318)  # for reproducibility
import tensorflow as tf
tf.set_random_seed(3318)
import random
random.seed(3318)
import sys
import re
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.externals import joblib
from keras.models import model_from_json

# config
testPath = sys.argv[1]
outputPath = sys.argv[2]
tagPath = "model/tags.npy"
tokenizerPath = "model/tokenizer.p"
modelDir = "model_keras"
numComponents = 200

# read testing data
print("\x1b[0;31;40m<I/O> Reading Testing Data\x1b[0m")
testData = []
tags = np.load(tagPath)
tag2Idx = {tags[i]: i for i in range(tags.shape[0])}
numTags = tags.shape[0]
tokenizer = pickle.load( open(tokenizerPath, "rb") )
with open(testPath, "r", encoding = "cp1252", errors = "replace") as fin:
    for lineNum, line in enumerate(fin):
        if lineNum > 0:
            print("\rProcessing - Line " + str(lineNum + 1) + "/1235" , end = "")
            splitedLine = line.rstrip().split(",")
            testData.append( " ".join(splitedLine[1:]) )
print("")

# tokenize
print("\x1b[0;31;40m<Keras> Generating TFIDF Matrix\x1b[0m")
testData = tokenizer.texts_to_matrix(testData, mode = "tfidf")
numWords = testData.shape[1]
print("tags.shape tags[0] |", tags.shape, tags[0])
print("len(tag2Idx) tag2Idx[\"HUMOUR\"] |", len(tag2Idx), tag2Idx["HUMOUR"])
print("testData.shape |", testData.shape)
print("numWords |", numWords)

# lsa
print("\x1b[0;31;40m<Sklearn> SVD - LSA\x1b[0m")
svd = TruncatedSVD(n_components = numComponents, n_iter = 10, random_state = 3318)
svd = joblib.load(modelDir + "/svd.pkl")
testData = svd.transform(testData)
print("testData.shape testData[0, :] |", testData.shape, testData[0, :])

print("\x1b[0;31;40m<Keras> Predicting\x1b[0m")
json_file = open(modelDir + "/model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(modelDir + "/model.h5")
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
predictions = model.predict(testData, verbose = 1)
print("\npredictions.shape |", predictions.shape)

# output
print("\x1b[0;31;40m<Output> Writing Result to CSV File\x1b[0m")
with open(outputPath, "w") as fout:
    fout.write("\"id\",\"tags\"\n")
    for i in range(predictions.shape[0]):
        curPredictions = predictions[i, :]
        fout.write("\"" + str(i) + "\",\"")
        first = True
        for j in range(numTags):
            if curPredictions[j] < 0.25: continue
            if first:
                fout.write(tags[j])
                first = False
            else:
                fout.write(" " + tags[j])
        if first:
            fout.write(tags[np.argmax(curPredictions)])
        fout.write("\"\n")

