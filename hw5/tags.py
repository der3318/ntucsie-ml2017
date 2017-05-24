from __future__ import print_function
import numpy as np
np.random.seed(3318)  # for reproducibility
import tensorflow as tf
tf.set_random_seed(3318)
import sys
import re
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
import pickle

# read training data
print("\x1b[0;31;40m<I/O> Reading Tags in Training Data\x1b[0m")
tags = set()
trainData = []
with open("../data/train_data.csv", "r", encoding = "cp1252", errors = "replace") as fin:
    for lineNum, line in enumerate(fin):
        if lineNum > 0:
            print("\rProcessing - Line " + str(lineNum + 1) + "/4965" , end = "")
            splitedLine = line.rstrip().split("\"")
            trainData.append( " ".join(splitedLine[2:]) )
            tags.update( list( (splitedLine[1]).split() ) )
print("")
with open("../data/test_data.csv", "r", encoding = "cp1252", errors = "replace") as fin:
    for lineNum, line in enumerate(fin):
        if lineNum > 0:
            print("\rProcessing - Line " + str(lineNum + 1) + "/1235" , end = "")
            splitedLine = line.rstrip().split(",")
            trainData.append( " ".join(splitedLine[1:]) )
tags = np.array([tag for tag in iter(tags)])
trainData = np.array(trainData)
print("\ntags | ", tags)
print("trainData.shape trainData[2] |", trainData.shape, trainData[2])

# save tags
print("\x1b[0;31;40m<I/O> Saving Tags to \"model/tags.npy\"\x1b[0m")
np.save("model/tags.npy", tags)

# indexing
print("\x1b[0;31;40m<Text> Indexing\x1b[0m")
tokenizer = Tokenizer(num_words = 50000)
tokenizer.fit_on_texts(trainData)
word2Idx = tokenizer.word_index
frequentWords = []
for word, i in word2Idx.items():
    if i < 20:  frequentWords.append( (word, i) )
print("len(word2Idx) word2Idx[\"living\"] |", len(word2Idx), word2Idx["living"])
print("frequentWords |", frequentWords)

# save index
print("\x1b[0;31;40m<I/O> Saving Tokenizer to \"model/\"\x1b[0m")
np.save("model/word2Idx.npy", word2Idx)
pickle.dump( tokenizer, open("model/tokenizer.p", "wb") )

