from __future__ import print_function
import numpy as np
np.random.seed(3318)  # for reproducibility
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix

batch_size = 128
nb_classes = 7
nb_epoch = 60
flag_flip = True
filters = []

# check load/semi flag
flag_load = True if("--load" in sys.argv) else False
flag_semi = True if("--semi" in sys.argv) else False
flag_semi_load = True if ("--semi_load" in sys.argv) else False

# input image dimensions
img_rows, img_cols = 48, 48
input_shape = (1, img_rows, img_cols)
# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# handle training data I/O
print("\x1b[0;31;40m<I/O> Reading Training Data\x1b[0m")
X_train = []
Y_train = []
L_train = []
with open(sys.argv[1], "r") as fin:
    for lineNum, line in enumerate(fin):
        if lineNum > 0:
            print("\rProcessing - Line " + str(lineNum + 1) + "/28710" , end = "")
            X_train.append( list( map( float, line.rstrip().split(",")[1].split() ) ) )
            oneHot = np.zeros(nb_classes)
            oneHot[int(line.rstrip().split(",")[0])] = 1.
            Y_train.append(oneHot)
            L_train.append( int(line.rstrip().split(",")[0]) )
X_train = np.array(X_train) / 255.
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
if flag_flip:
    X_train_flip = X_train[:, :, :, ::-1]
    X_train = np.concatenate( (X_train, X_train_flip), axis = 0 )
rnd_perm = np.random.permutation(X_train.shape[0])
X_train = X_train[rnd_perm, :, :, :]
Y_train = np.array(Y_train)
L_train = np.array(L_train)
if flag_flip:
    Y_train = np.concatenate( (Y_train, Y_train), axis = 0 )[rnd_perm, :]
    L_train = np.concatenate( (L_train, L_train), axis = 0 )[rnd_perm]
else:
    Y_train = Y_train[rnd_perm, :]
    L_train = L_train[rnd_perm]
print("\nShape of X_train Y_train |", X_train.shape, Y_train.shape)

# keras setup
print("\x1b[0;31;40m<Training> Keras - CNN and DNN\x1b[0m")
model = Sequential()
# cnn1
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=input_shape))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))
# cnn2
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))
# cnn3
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(BatchNormalization(mode=0, axis=1))
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
model.add(Activation('sigmoid'))
model.add(Dropout(0.48))
# output
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
if not flag_load:
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)
    #print("\x1b[0;31;40m<Save> Saving Keras Model\x1b[0m")
    #model_json = model.to_json()
    #with open("keras_load/model.json", "w") as json_file:
        #json_file.write(model_json)
        #model.save_weights("keras_load/model.h5")
else:
    print("\x1b[0;31;40m<Skip> Loading Keras Model\x1b[0m")
    json_file = open("keras_load/model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("keras_load/model.h5")
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    filters = model.layers[0].get_weights()
score = model.evaluate(X_train, Y_train, verbose=0)
print("Score and Accuracy |", score[0], score[1])
# confusion matrix
predictions = model.predict_classes(X_train, verbose = 0)
print( "Confusion Matrix |", confusion_matrix(L_train, predictions) )

# handle testing data I/O
print("\x1b[0;31;40m<I/O> Reading Testing Data\x1b[0m")
X_test = []
with open(sys.argv[2], "r") as fin:
    for lineNum, line in enumerate(fin):
        if lineNum > 0:
            print("\rProcessing - Line " + str(lineNum + 1) + "/7179" , end = "")
            X_test.append( list( map( float, line.rstrip().split(",")[1].split() ) ) )
X_test = np.array(X_test) / 255.
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
print("\nShape of X_test |", X_test.shape)

# semi
new_index = []
new_label = []
if flag_semi:
    print("\x1b[0;31;40m<Semi-Supervised Learning> Adding Confident Testing Data and Training\x1b[0m")
    predictions = model.predict_proba(X_test)
    for i in range(X_test.shape[0]):
        label = np.argmax(predictions[i])
        if predictions[i][label] > 0.88:
            new_index.append(i)
            new_label.append(label)
    X_train_new = np.concatenate( (X_train, X_test[new_index, :, :, :]), axis = 0 )
    X_train_new = np.concatenate( (X_train_new, X_test[new_index, :, :, ::-1]), axis = 0 )
    rnd_perm = np.random.permutation(X_train_new.shape[0])
    X_train_new = X_train_new[rnd_perm, :, :, :]
    oneHots = np.zeros([len(new_index) * 2, nb_classes])
    oneHots[range(len(new_index) * 2), np.append(new_label, new_label)] = 1.
    Y_train_new = np.concatenate( (Y_train, oneHots), axis = 0 )[rnd_perm, :]
    print("\nShape of X_train_new Y_train_new |", X_train_new.shape, Y_train_new.shape)
    if not flag_semi_load:
        model.fit(X_train_new, Y_train_new, batch_size=batch_size, nb_epoch=nb_epoch // 4, verbose=1)
        #print("\x1b[0;31;40m<Save> Saving Keras Model\x1b[0m")
        #model_json = model.to_json()
        #with open("keras_load/model_semi.json", "w") as json_file:
            #json_file.write(model_json)
            #model.save_weights("keras_load/model_semi.h5")
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

# saliency map of image 0
modify = []
for row in range(img_rows):
    for col in range(img_cols):
        origin_modify = np.copy(X_train[1, :, :, :])
        origin_modify[:, row, col] = origin_modify[:, row, col] + (1 / 255)
        modify.append(origin_modify)
modify = np.array(modify)
score = model.evaluate(X_train[[1], :, :, :], Y_train[[1], :], verbose = 0)[0]
score_modify = []
for i in range(img_rows * img_cols):
    score_modify.append(model.evaluate(modify[[i], :, :, :], Y_train[[1], :], verbose = 0)[0])
delta = np.zeros([img_rows, img_cols])
for i in range(img_rows * img_cols):
    (row, col) = (i // img_cols, i % img_cols)
    delta[row, col] = score - score_modify[i]
with open("saliency.txt", "w") as fout:
    for i in range(img_rows * img_cols):
        (row, col) = (i // img_cols, i % img_cols)
        fout.write(str(X_train[1, 0, row, col]) + " ")
    fout.write("\n")
    for i in range(img_rows * img_cols):
        (row, col) = (i // img_cols, i % img_cols)
        fout.write(str(delta[row, col]) + " ")
    fout.write("\n")

# print filters
filters = filters[0].reshape(64, 3, 3)
with open("weights.txt", "w") as fout:
    for i in range(filters.shape[0]):
        for r in range(filters.shape[1]):
            for c in range(filters.shape[2]):
                fout.write(str(filters[i, r, c]) + " ")
        fout.write("\n")


