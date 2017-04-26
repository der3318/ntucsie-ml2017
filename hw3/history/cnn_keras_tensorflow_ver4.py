from __future__ import print_function
import numpy as np
np.random.seed(3318)  # for reproducibility
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json

batch_size = 64
nb_classes = 7
nb_epoch = 40
flag_flip = True

# check load/semi flag
flag_load = True if("--load" in sys.argv) else False
flag_semi = True if("--semi" in sys.argv) else False
flag_semi_load = True if ("--semi_load" in sys.argv) else False

# input image dimensions
img_rows, img_cols = 48, 48
input_shape = (img_rows, img_cols, 1)
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# handle training data I/O
print("\x1b[0;31;40m<I/O> Reading Training Data\x1b[0m")
X_train = []
Y_train = []
with open(sys.argv[1], "r") as fin:
    for lineNum, line in enumerate(fin):
        if lineNum > 0:
            print("\rProcessing - Line " + str(lineNum + 1) + "/28710" , end = "")
            X_train.append( list( map( float, line.rstrip().split(",")[1].split() ) ) )
            oneHot = np.zeros(nb_classes)
            oneHot[int(line.rstrip().split(",")[0])] = 1.
            Y_train.append(oneHot)
X_train = np.array(X_train) / 255.
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
if flag_flip:
    X_train_flip = X_train[:, :, ::-1, :]
    X_train = np.concatenate( (X_train, X_train_flip), axis = 0 )
rnd_perm = np.random.permutation(X_train.shape[0])
X_train = X_train[rnd_perm, :, :, :]
Y_train = np.array(Y_train)
if flag_flip:
    Y_train = np.concatenate( (Y_train, Y_train), axis = 0 )[rnd_perm, :]
else:
    Y_train = Y_train[rnd_perm, :]
print("\nShape of X_train Y_train |", X_train.shape, Y_train.shape)

# keras setup
print("\x1b[0;31;40m<Training> Keras - CNN and DNN\x1b[0m")
model = Sequential()
# cnn1
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=input_shape))
model.add(Activation('relu'))
# cnn2
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))
# cnn3
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))
# dnn1
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.48))
# dnn2
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.48))
# output
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
if not flag_load:
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)
    print("\x1b[0;31;40m<Save> Saving Keras Model\x1b[0m")
    model_json = model.to_json()
    with open("keras_load/model.json", "w") as json_file:
        json_file.write(model_json)
        model.save_weights("keras_load/model.h5")
else:
    print("\x1b[0;31;40m<Skip> Loading Keras Model\x1b[0m")
    json_file = open("keras_load/model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("keras_load/model.h5")
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
score = model.evaluate(X_train, Y_train, verbose=0)
print("Score and Accuracy |", score[0], score[1])

# handle testing data I/O
print("\x1b[0;31;40m<I/O> Reading Testing Data\x1b[0m")
X_test = []
with open(sys.argv[2], "r") as fin:
    for lineNum, line in enumerate(fin):
        if lineNum > 0:
            print("\rProcessing - Line " + str(lineNum + 1) + "/7179" , end = "")
            X_test.append( list( map( float, line.rstrip().split(",")[1].split() ) ) )
X_test = np.array(X_test) / 255.
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
print("\nShape of X_test |", X_test.shape)

# semi
new_index = []
new_label = []
if flag_semi:
    print("\x1b[0;31;40m<Semi-Supervised Learning> Adding Confident Testing Data and Training\x1b[0m")
    predictions = model.predict_proba(X_test)
    for i in range(X_test.shape[0]):
        label = np.argmax(predictions[i])
        if predictions[i][label] > 0.85:
            new_index.append(i)
            new_label.append(label)
    X_train_new = np.concatenate( (X_train, X_test[new_index, :, :, :]), axis = 0 )
    X_train_new = np.concatenate( (X_train_new, X_test[new_index, :, ::-1, :]), axis = 0 )
    rnd_perm = np.random.permutation(X_train_new.shape[0])
    X_train_new = X_train_new[rnd_perm, :, :, :]
    oneHots = np.zeros([len(new_index) * 2, nb_classes])
    oneHots[range(len(new_index) * 2), np.append(new_label, new_label)] = 1.
    Y_train_new = np.concatenate( (Y_train, oneHots), axis = 0 )[rnd_perm, :]
    print("\nShape of X_train_new Y_train_new |", X_train_new.shape, Y_train_new.shape)
    if not flag_semi_load:
        model.fit(X_train_new, Y_train_new, batch_size=batch_size, nb_epoch=nb_epoch // 4, verbose=1)
        print("\x1b[0;31;40m<Save> Saving Keras Model\x1b[0m")
        model_json = model.to_json()
        with open("keras_load/model_semi.json", "w") as json_file:
            json_file.write(model_json)
            model.save_weights("keras_load/model_semi.h5")
    else:
        print("\x1b[0;31;40m<Skip> Loading Keras Model\x1b[0m")
        json_file = open("keras_load/model_semi.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("keras_load/model_semi.h5")
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    score = model.evaluate(X_train_new, Y_train_new, verbose=0)
    print("Score and Accuracy |", score[0], score[1])

# output
print("\x1b[0;31;40m<Testing> Keras - Predicting Classes\x1b[0m")
predictions = model.predict_classes(X_test)
with open(sys.argv[3], "w") as fout:
    fout.write('id,label\n')
    for i in range(X_test.shape[0]):
        fout.write(str(i) + "," + str(predictions[i]) + "\n")
print("\n\x1b[0;32;40m<Done> Output File Available\x1b[0m")

