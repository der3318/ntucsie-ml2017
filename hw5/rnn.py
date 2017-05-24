from __future__ import print_function
import numpy as np
np.random.seed(3318)  # for reproducibility
import tensorflow as tf
tf.set_random_seed(3318)
import random
random.seed(3318)
import sys
import re
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

def f1score(y_true, y_pred):
    thresh = 0.5
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))

# config
trainPath = "../data/train_data.csv"
tagPath = "model_rnn/tags.npy"
idxPath = "model_rnn/word2Idx.npy"
embeddingPath = "model_rnn/glove.6B.50d.txt"
modelDir = "model_keras_rnn"
numWords = 3000
seqLen = 50
batchSize = 32
numEpochs = 25

# read tags and training data
print("\x1b[0;31;40m<I/O> Reading Tags, Word Index and Training Data\x1b[0m")
trainData = []
trainLabel = []
tags = np.load(tagPath)
tag2Idx = {tags[i]: i for i in range(tags.shape[0])}
numTags = tags.shape[0]
tagIdxList = [[] for i in range(numTags)]
word2Idx = np.load(idxPath).item()
with open(trainPath, "r", encoding = "cp1252", errors = "replace") as fin:
    for lineNum, line in enumerate(fin):
        if lineNum > 0:
            print("\rProcessing - Line " + str(lineNum + 1) + "/4965" , end = "")
            splitedLine = line.rstrip().split("\"")
            lineData = []
            for word in re.sub( "[^a-zA-Z ]+", " ", " ".join(splitedLine[2:]).replace(",", " ") ).lower().split():
                if word in word2Idx and word2Idx[word] < numWords:  lineData.append( word2Idx[word] )
            trainData.append(lineData)
            lineLabel = np.zeros(tags.shape[0])
            for tag in splitedLine[1].split():
                lineLabel[ tag2Idx[tag]  ] = 1.
                tagIdxList[ tag2Idx[tag] ].append(lineNum - 1)
            trainLabel.append(lineLabel)
trainData = sequence.pad_sequences(trainData, maxlen = seqLen)
trainLabel = np.array(trainLabel)
tagIdxList = np.array(tagIdxList)
np.save( "model/tagsize.npy", np.array([len(tagIdxList[i]) / trainData.shape[0] for i in range(numTags)]) )
print("\ntags.shape tags[0] |", tags.shape, tags[0])
print("len(tag2Idx) tag2Idx[\"HUMOUR\"] |", len(tag2Idx), tag2Idx["HUMOUR"])
print("trainData.shape trainData[2] |", trainData.shape, trainData[2])
print("trainLabel.shape trainLabel[0, :] |", trainLabel.shape, trainLabel[0, :])
print("tagIdxList.shape[0] |", tagIdxList.shape[0])
print("tagSize |", [len(tagIdxList[i]) for i in range(numTags)])
print("numWords |", numWords)

# read embedding
print("\x1b[0;31;40m<I/O> Reading Glove Embedding\x1b[0m")
embedding = {}
with open(embeddingPath, "r") as fin:
    for line in fin:
        embedding[ line.split()[0] ] = np.asarray(line.split()[1:], dtype = "float32")
embeddingDim = embedding["living"].shape[0]
print("len(embedding) embedding[\"living\"] |", len(embedding), embedding["living"])

# build embedding matrix
print("\x1b[0;31;40m<Init> Building Embedding Matrix\x1b[0m")
embeddingMat = np.zeros( (numWords, embeddingDim) )
for word, i in word2Idx.items():
    if i >= numWords:   continue
    wordEmb = embedding.get(word)
    if wordEmb is not None: embeddingMat[i, :] = wordEmb
print("embeddingMat.shape embeddingMat[word2Idx[\"living\"], :] |", embeddingMat.shape, embeddingMat[word2Idx["living"], :])

# keras model
print("\x1b[0;31;40m<Keras> RNN Training\x1b[0m")
model = Sequential()
model.add( Embedding(numWords, embeddingDim, weights = [embeddingMat], input_length = seqLen, trainable = False) )
model.add( LSTM(128, dropout = 0.2, recurrent_dropout = 0.2) )
model.add( Dropout(0.2) )
model.add( Dense(64) )
model.add( Dropout(0.2) )
model.add( Activation("relu") )
model.add( Dense(numTags, activation = "sigmoid") )
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = [f1score])
model.summary()
if "--train" in sys.argv:
    model.fit(trainData, trainLabel, batch_size = batchSize, epochs = numEpochs, validation_split = 0.2)
    score = model.evaluate(trainData, trainLabel, batch_size = batchSize)
    print("\nscore |", score)

# save model
if "--train" in sys.argv:
    print("\x1b[0;31;40m<Save> Saving Keras Model\x1b[0m")
    model_json = model.to_json()
    with open(modelDir + "/model.json", "w") as json_file:
        json_file.write(model_json)
        model.save_weights(modelDir + "/model.h5")

