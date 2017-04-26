from __future__ import print_function
import numpy as np
np.random.seed(3318)  # for reproducibility
import sys
from keras.models import model_from_json

# handle testing data I/O
print("\x1b[0;31;40m<I/O> Reading Testing Data\x1b[0m")
X_test = []
with open(sys.argv[1], "r") as fin:
    for lineNum, line in enumerate(fin):
        if lineNum > 0:
            print("\rProcessing - Line " + str(lineNum + 1) + "/7179" , end = "")
            X_test.append( list( map( float, line.rstrip().split(",")[1].split() ) ) )
X_test = np.array(X_test) / 255.
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
print("\nShape of X_test |", X_test.shape)

# loading model
print("\x1b[0;31;40m<Loading> Predicions of 5 Keras Models\x1b[0m")
model_predictions = []
paths = ["keras_load_ver1", "keras_load_ver2", "keras_load_ver6", "keras_load_ver4", "keras_load_ver5"]
for path in paths:
    json_file = open(path + "/model_semi.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(path + "/model_semi.h5")
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    predictions = model.predict_classes(X_test)
    model_predictions.append(predictions)
    print("")

# ensemble
print("\x1b[0;31;40m<Ensemble> Keras - Performing Ensemble\x1b[0m")
with open(sys.argv[2], "w") as fout:
    fout.write('id,label\n')
    for i in range(X_test.shape[0]):
        votes = np.random.uniform(0, 0.5, 7)
        for pred in model_predictions:
            votes[ pred[i] ] += 1.
        label = np.argmax(votes)
        fout.write(str(i) + "," + str(label) + "\n")
print("\x1b[0;32;40m<Done> Output File Available\x1b[0m")


