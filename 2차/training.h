#pragma once
#include "structure.h"

#define DIMENSION 13

void setData(Data data, Node* layer0);	//set up the bottom layer with training data
void training(Data input, vector<Node*> layers, vector<MAT*> W, double** relation, double threshold);	//input Training data, hidden layers, Weight Matrix return Matrix
double testing(Data input, vector<Node*> layers, vector<MAT*>);
