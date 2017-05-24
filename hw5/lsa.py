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
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, Activation
from keras.layers import LSTM, Concatenate, Input, concatenate
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.models import model_from_json
from keras.layers.merge import Concatenate
import keras.backend as K
from sklearn.decomposition import TruncatedSVD
from sklearn.externals import joblib

def f1score(y_true, y_pred):
    thresh = 0.25
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))

# config
trainPath = "../data/train_data.csv"
testPath = "../data/test_data.csv"
outputPath = "out.csv"
tagPath = "model/tags.npy"
tokenizerPath = "model/tokenizer.p"
embeddingPath = "model/glove.6B.50d.txt"
modelDir = "model_keras"
numComponents = 200
batchSize = 128
numEpochs = 25

# read tags and training data
print("\x1b[0;31;40m<I/O> Reading Tags, Word Index and Training Data\x1b[0m")
trainData = []
trainLabel = []
tags = np.load(tagPath)
tag2Idx = {tags[i]: i for i in range(tags.shape[0])}
numTags = tags.shape[0]
tagIdxList = [[] for i in range(numTags)]
tokenizer = pickle.load( open(tokenizerPath, "rb") )
word2Idx = tokenizer.word_index
with open(trainPath, "r", encoding = "cp1252", errors = "replace") as fin:
    for lineNum, line in enumerate(fin):
        if lineNum > 0:
            print("\rProcessing - Line " + str(lineNum + 1) + "/4965" , end = "")
            splitedLine = line.rstrip().split("\"")
            trainData.append( " ".join(splitedLine[2:]) )
            lineLabel = np.zeros(tags.shape[0])
            for tag in splitedLine[1].split():
                lineLabel[ tag2Idx[tag]  ] = 1.
                tagIdxList[ tag2Idx[tag] ].append(lineNum - 1)
            trainLabel.append(lineLabel)
trainLabel = np.array(trainLabel)
tagIdxList = np.array(tagIdxList)
print("")

# read testing data
print("\x1b[0;31;40m<I/O> Reading Testing Data\x1b[0m")
with open(testPath, "r", encoding = "cp1252", errors = "replace") as fin:
    for lineNum, line in enumerate(fin):
        if lineNum > 0:
            print("\rProcessing - Line " + str(lineNum + 1) + "/1235" , end = "")
            splitedLine = line.rstrip().split(",")
            trainData.append( " ".join(splitedLine[1:]) )
print("")

# tokenize
print("\x1b[0;31;40m<Keras> Generating TFIDF Matrix\x1b[0m")
dataMatrix = tokenizer.texts_to_matrix(trainData, mode = "tfidf")
trainData = dataMatrix[:trainLabel.shape[0], :]
testData = dataMatrix[trainLabel.shape[0]:, :]
numWords = dataMatrix.shape[1]
np.save( "model/tagsize.npy", np.array([len(tagIdxList[i]) / trainData.shape[0] for i in range(numTags)]) )
print("tags.shape tags[0] |", tags.shape, tags[0])
print("len(tag2Idx) tag2Idx[\"HUMOUR\"] |", len(tag2Idx), tag2Idx["HUMOUR"])
print("trainData.shape trainData[0, :] |", trainData.shape, trainData[0, :])
print("trainLabel.shape trainLabel[0, :] |", trainLabel.shape, trainLabel[0, :])
print("testData.shape |", testData.shape)
print("tagIdxList.shape[0] |", tagIdxList.shape[0])
print("tagSize |", [len(tagIdxList[i]) for i in range(numTags)])
print("numWords |", numWords)

# lsa
print("\x1b[0;31;40m<Sklearn> SVD - LSA\x1b[0m")
svd = TruncatedSVD(n_components = numComponents, n_iter = 10, random_state = 3318)
if "--train_svd" in sys.argv:
    svd.fit(dataMatrix)
    joblib.dump(svd, modelDir + "/svd.pkl")
else:
    svd = joblib.load(modelDir + "/svd.pkl")
trainData = svd.transform(trainData)
testData = svd.transform(testData)
print("trainData.shape trainData[0, :] |", trainData.shape, trainData[0, :])
print("testData.shape testData[0, :] |", testData.shape, testData[0, :])

# dnn
print("\x1b[0;31;40m<Keras> DNN Training\x1b[0m")
mainInput = Input(shape = (numComponents,), dtype = "float32", name = "mainInput")
dense1 = Dense(256) (mainInput)
dropout1 = Dropout(0.2) (dense1)
act1 = Activation("relu") (dropout1)
dense2 = Dense(256) (act1)
dropout2 = Dropout(0.2) (dense2)
act2 = Activation("relu") (dropout2)
dense3 = Dense(128) (act2)
dropout3 = Dropout(0.2) (dense3)
act3 = Activation("relu") (dropout3)
dense4 = Dense(128) (act3)
dropout4 = Dropout(0.2) (dense4)
act4 = Activation("relu") (dropout4)
mainOutput = Dense(numTags, activation = "sigmoid") (act4)
model = Model(inputs = [mainInput], outputs = [mainOutput])
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = [f1score])
model.summary()
if "--train_dnn" in sys.argv:
    model.fit(trainData, trainLabel, batch_size = batchSize, epochs = numEpochs, validation_split = 0.0)
    score = model.evaluate(trainData, trainLabel, batch_size = batchSize)
    print("\nscore |", score)
    model_json = model.to_json()
    with open(modelDir + "/model.json", "w") as json_file:
        json_file.write(model_json)
        model.save_weights(modelDir + "/model.h5")

# predict
if "--test" not in sys.argv:    sys.exit()
json_file = open(modelDir + "/model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(modelDir + "/model.h5")
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
predictions = model.predict(testData, verbose = 1)
print("\npredictions.shape |", predictions.shape)

# threshold
print("\x1b[0;31;40m<Calc> Computing Threshold\x1b[0m")
tagSize = np.load("model/tagsize.npy")
threshold = np.zeros(numTags)
for tagIdx in range(numTags):
    arg = np.argsort(predictions[:, tagIdx])
    threshold[tagIdx] = predictions[arg[-int(tagSize[tagIdx] * testData.shape[0] * 1.7)], tagIdx]
print("tagSize |", tagSize)
print("threshold |", threshold)

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
            #if curPredictions[j] < threshold[j]: continue
            if first:
                fout.write(tags[j])
                first = False
            else:
                fout.write(" " + tags[j])
        if first:
            fout.write(tags[np.argmax(curPredictions)])
        fout.write("\"\n")

