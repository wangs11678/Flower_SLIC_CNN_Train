#ifndef PREDICT_H_INCLUDED
#define PREDICT_H_INCLUDED

#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <iostream>
#include <vector>
#include "liblinear/linear.h"

std::vector<int> SVM_predict(int argc, char **argv, char *testFile, char *modelFile, char *resultFile);

#endif // PREDICT_H_INCLUDED
