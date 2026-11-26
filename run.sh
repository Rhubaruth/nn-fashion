#!/bin/bash
## change this file to your needs

echo "Adding some modules"


# module add gcc-10.2


echo "#################"
echo "    COMPILING    "
echo "#################"

## dont forget to use comiler optimizations (e.g. -O3 or -Ofast)
g++ -Wall -std=c++17 -Ofast -march=native --sanitize=address src/main.cpp src/layers.cpp src/load_data.cpp -o network


echo "#################"
echo "     RUNNING     "
echo "#################"

## use nice to decrease priority in order to comply with aisa rules
## https://www.fi.muni.cz/tech/unix/computation.html.en
## especially if you are using multiple cores
# ./network

touch train_predictions.csv
touch test_predictions.csv

# for debugging
ASAN=$(ldd ./network | awk '/libasan/ {print $3; exit}')
echo "$ASAN"
time LD_PRELOAD="$ASAN" ./network

python evaluator/evaluate.py train_predictions.csv data/fashion_mnist_train_labels.csv  
python evaluator/evaluate.py test_predictions.csv data/fashion_mnist_test_labels.csv  
