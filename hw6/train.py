import sys
import numpy as np
np.random.seed(3318)  # for reproducibility
import tensorflow as tf
tf.set_random_seed(3318) # for repreducibility
from keras.layers import Input, Dense, Dropout, Embedding, Flatten, dot, add, concatenate
from keras.models import Model
from keras.models import model_from_json
from keras import regularizers

# config
trainPath = "data/train.csv"
numPath = "model/num.npy"
userPath = "model/userMat.npy"
moviePath = "model/movieMat.npy"
modelDir = "model_keras"
batchSize = 128
numEpochs = 12

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
print("\x1b[0;31;40m<I/O> Reading Training Data\x1b[0m")
trainData = []
trainLabel = []
with open(trainPath) as fin:
    for lineNum, line in enumerate(fin):
        if (lineNum + 1) % 3318 == (num["train"] + 1) % 3318:
            print("\rProcessing - Line " + str(lineNum + 1) + "/" + str(num["train"] + 1), end = "")
        if lineNum == 0:    continue
        lineData = list( map( int, line.rstrip().split(",") ) )
        trainData.append([ lineData[1], lineData[2] ])
        trainLabel.append([lineData[3]])
trainData = np.array(trainData)
trainLabel = np.array(trainLabel)
rndPerm = np.random.permutation(trainData.shape[0])
trainData = trainData[rndPerm, :]
trainLabel = trainLabel[rndPerm, :]
print("\ntrainData.shape trainData[0, :] |", trainData.shape, trainData[0, :])
print("trainLabel.shape trainLabel[0, :] |", trainLabel.shape, trainLabel[0, :])

# model
print("\x1b[0;31;40m<Keras> DNN Training\x1b[0m")
if "--train" not in sys.argv:   sys.exit()
## inputs
userInput = Input( shape = (userMat.shape[1] - num["occupation"],) )
occuInput = Input( shape = (num["occupation"],) )
movieInput = Input( shape = (movieMat.shape[1] - num["tag"],) )
tagInput = Input( shape = (num["tag"],) )
userIdInput = Input( shape = (1,) )
movieIdInput = Input( shape = (1,) )
## embedding
occuVec = Dense(8, activation = "linear")(occuInput)
tagVec = Dense(8, activation = "linear")(tagInput)
userEmb = Embedding(num["user"], 8, input_length = 1, trainable = True)(userIdInput)
userVec = Flatten()(userEmb)
movieEmb = Embedding(num["movie"], 8, input_length = 1, trainable = True)(movieIdInput)
movieVec = Flatten()(movieEmb)
dot1 = dot([occuVec, tagVec], axes = -1)
dot2 = dot([occuVec, userVec], axes = -1)
dot3 = dot([occuVec, movieVec], axes = -1)
dot4 = dot([tagVec, userVec], axes = -1)
dot5 = dot([tagVec, movieVec], axes = -1)
dot6 = dot([userVec, movieVec], axes = -1)
## bias
userBiasEmb = Embedding(num["user"], 1, input_length = 1, trainable = True)(userIdInput)
userBias = Flatten()(userBiasEmb)
movieBiasEmb = Embedding(num["movie"], 1, input_length = 1, trainable = True)(movieIdInput)
movieBias = Flatten()(movieBiasEmb)
## concat
concat = concatenate([userInput, movieInput, dot1, dot2, dot3, dot4, dot5, dot6])
denseOutput = Dense(1, activation = "linear")(concat)
## add
output = add([denseOutput, userBias, movieBias])
## model
model = Model(inputs = [userInput, occuInput, movieInput, tagInput, userIdInput, movieIdInput], outputs = output)
model.compile(loss = "mean_squared_error", optimizer = "adam")
model.summary()
model.fit(
    [   userMat[trainData[:, 0], :(userMat.shape[1] - num["occupation"])],
        userMat[trainData[:, 0], (userMat.shape[1] - num["occupation"]):],
        movieMat[trainData[:, 1], :(movieMat.shape[1] - num["tag"])],
        movieMat[trainData[:, 1], (movieMat.shape[1] - num["tag"]):],
        trainData[ :, [0] ],
        trainData[ :, [1] ] ],
    trainLabel, batch_size = batchSize, epochs = numEpochs, validation_split = 0.0)
score = model.evaluate(
    [   userMat[trainData[:, 0], :(userMat.shape[1] - num["occupation"])],
        userMat[trainData[:, 0], (userMat.shape[1] - num["occupation"]):],
        movieMat[trainData[:, 1], :(movieMat.shape[1] - num["tag"])],
        movieMat[trainData[:, 1], (movieMat.shape[1] - num["tag"]):],
        trainData[ :, [0] ],
        trainData[ :, [1] ] ],
    trainLabel)
print("\nscore |", score)

# save model
print("\x1b[0;31;40m<Keras> Saving Keras Model\x1b[0m")
model_json = model.to_json()
with open(modelDir + "/model.json", "w") as json_file:
    json_file.write(model_json)
    model.save_weights(modelDir + "/model.h5")

