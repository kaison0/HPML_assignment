#!/bin/bash
set -x

CXX_FLAGS="-std=c++11"
CUDA_FLAGS="-arch=sm_60"

echo "Compiling Q1.cpp"
g++ $CXX_FLAGS -o Q1 Q1.cpp

echo "Compiling Q2.cu"
nvcc $CUDA_FLAGS -o Q2 Q2.cu

echo "Compiling Q3.cu"
nvcc $CUDA_FLAGS -o Q3 Q3.cu

./Q1 1
./Q1 5
./Q1 10
./Q1 50
./Q1 100

./Q2 1 1
./Q2 1 2
./Q2 1 3
./Q2 5 1
./Q2 5 2 
./Q2 5 3
./Q2 10 1
./Q2 10 2
./Q2 10 3
./Q2 50 1
./Q2 50 2
./Q2 50 3
./Q2 100 1
./Q2 100 2
./Q2 100 3

./Q3 1 1
./Q3 1 2
./Q3 1 3
./Q3 5 1
./Q3 5 2 
./Q3 5 3
./Q3 10 1
./Q3 10 2
./Q3 10 3
./Q3 50 1
./Q3 50 2
./Q3 50 3
./Q3 100 1
./Q3 100 2
./Q3 100 3

