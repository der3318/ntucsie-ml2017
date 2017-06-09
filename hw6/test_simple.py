import os
import sys
import numpy as np
np.random.seed(3318)  # for reproducibility
import tensorflow as tf
tf.set_random_seed(3318) # for repreducibility
from keras.layers import Input, Dense, Dropout, Embedding, Flatten, dot, add
from keras.models import Model
from keras.models import model_from_json

# config
testPath = os.path.join(sys.argv[1], "test.csv")
outputPath = sys.argv[2]
numPath = "model/num.npy"
userPath = "model/userMat.npy"
moviePath = "model/movieMat.npy"
modelDir = "model_keras_simple"

# read numbers
print("\x1b[0;31;40m<I/O> Reading Numbers, Users and Movies\x1b[0m")
num = np.load(numPath).item()
userMat = np.load(userPath)
movieMat = np.load(moviePath)
print("num[\"train\"] |", num["train"])
print("num[\"test\"] |", num["test"])
print("num[\"user\"] |", num["user"])
print("num[\"movie\"] |", num["movie"])
for i in range(6):
    print("num[\"rate" + str(i) + "\"] |", num["rate" + str(i)])
print("userMat.shape |", userMat.shape)
print("movieMat.shape |", movieMat.shape)

# read training data
print("\x1b[0;31;40m<I/O> Reading Testing Data\x1b[0m")
testID = []
testData = []
with open(testPath) as fin:
    for lineNum, line in enumerate(fin):
        if (lineNum + 1) % 3318 == (num["test"] + 1) % 3318:
            print("\rProcessing - Line " + str(lineNum + 1) + "/" + str(num["test"] + 1), end = "")
        if lineNum == 0:    continue
        lineData = list( map( int, line.rstrip().split(",") ) )
        testID.append(lineData[0])
        testData.append([ lineData[1], lineData[2] ])
testData = np.array(testData)
print("\ntestData.shape testData[0, :] |", testData.shape, testData[0, :])

# model
print("\x1b[0;31;40m<Keras> Loading Model\x1b[0m")
json_file = open(modelDir + "/model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(modelDir + "/model.h5")
model.compile(loss = "mean_squared_error", optimizer = "adam")
predictions = model.predict(
    [   testData[ :, [0] ],
        testData[ :, [1] ] ],
    verbose = 1)
print("\npredictions.shape predictions[0, :] |", predictions.shape, predictions[0, :])

# output
print("\x1b[0;31;40m<Output> Writing Result to CSV File\x1b[0m")
with open(outputPath, "w") as fout:
    fout.write("TestDataID,Rating\n")
    for i in range(predictions.shape[0]):
        value = predictions[i, 0]
        if value > 5:   value = 5
        if value < 1:   value = 1
        fout.write(str(testID[i]) + "," + str(value) + "\n")

