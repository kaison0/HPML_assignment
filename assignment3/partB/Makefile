###
### For COMS E6998 Spring 2024
### Instructor: Kaoutar El Maghraoui
### Makefile for CUDA1 assignment
### By Wim Bohm, Waruna Ranasinghe, and Louis Rabiet
### Created: 2011-01-27 DVN
### Last Modified: Nov 2014 WB, WR, LR
###

SDK_INSTALL_PATH := /usr/local/cuda
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
#INCLUDES  :=  -I$(SDK_INSTALL_PATH)/include -I$(SDK_INSTALL_PATH)/samples/common/inc
OPTIONS   :=  -O3 
#--maxrregcount=100 --ptxas-options -v 

EXECS :=  Q1 Q2 Q3
all:$(EXECS)

#######################################################################
clean:
	rm -f $(EXECS) *.o

timer.o : timer.cu timer.h
	${NVCC} $< -c -o $@ $(OPTIONS)


#######################################################################
Q1.O : Q1.cpp
	${NVCC} $< -c -o $@ $(OPTIONS)

Q1 : Q1.o timer.o
	${NVCC} $^ -o $@ $(LIB) $(OPTIONS)


#######################################################################
Q2.o : Q2.cu
	${NVCC} $< -c -o $@ $(OPTIONS)

Q2 : Q2.o timer.o
	${NVCC} $^ -o $@ $(LIB) $(OPTIONS)


#######################################################################
Q3.o : Q3.cu 
	${NVCC} $< -c -o $@ $(OPTIONS)

Q3 : Q3.o timer.o
	${NVCC} $^ -o $@ $(LIB) $(OPTIONS)





