#!/bin/bash

cp test.csv test100.csv -rf

cp history/cnn_keras_tensorflow_ver1.py cnn.py -rf
mkdir -p keras_load
# python3 cnn.py train.csv test.csv out.csv --semi
python3 cnn.py $1 test100.csv tmp.csv --semi
rm keras_load_ver1 -rf
cp keras_load keras_load_ver1 -rf

cp history/cnn_keras_tensorflow_ver2.py cnn.py -rf
mkdir -p keras_load
python3 cnn.py $1 test100.csv tmp.csv --semi
rm keras_load_ver2 -rf
cp keras_load keras_load_ver2 -rf

cp history/cnn_keras_tensorflow_ver3.py cnn.py -rf
mkdir -p keras_load
python3 cnn.py $1 test100.csv tmp.csv --semi
rm keras_load_ver3 -rf
cp keras_load keras_load_ver3 -rf

cp history/cnn_keras_tensorflow_ver4.py cnn.py -rf
mkdir -p keras_load
python3 cnn.py $1 test100.csv tmp.csv --semi
rm keras_load_ver4 -rf
cp keras_load keras_load_ver4 -rf

cp history/cnn_keras_tensorflow_ver5.py cnn.py -rf
mkdir -p keras_load
python3 cnn.py $1 test100.csv tmp.csv --semi
rm keras_load_ver5 -rf
cp keras_load keras_load_ver5 -rf

cp history/cnn_keras_tensorflow_ver6.py cnn.py -rf
mkdir -p keras_load
python3 cnn.py $1 test100.csv tmp.csv --semi
rm keras_load_ver6 -rf
cp keras_load keras_load_ver6 -rf

