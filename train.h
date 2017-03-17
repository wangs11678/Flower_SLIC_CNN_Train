#ifndef TRAIN_H_INCLUDED
#define TRAIN_H_INCLUDED

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <iostream>
#include "liblinear/linear.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

void SVM_train(int argc, char **argv, char* trainFile, char* modelFile);

#endif // TRAIN_H_INCLUDED
