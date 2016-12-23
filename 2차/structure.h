#pragma once
#include <vector>

#define DIMENSION 13	//input vector dimension
#define CLASS_NUM 2	//number of classes

using namespace std;

typedef struct _data {
	double x[DIMENSION];
	char r;	//0 or 1
}Data;

typedef vector<Data> DataSet;	//data set

typedef struct _matrix {
	int row;	
	int col;
	double **matrix;	//
}MAT;

typedef struct _node {
	double val;	//activation value
	char output;	//real output of the node
}Node;

/*functions for layer*/
Node* makeLayer(int n); //create a hidden layer which has n nodes
void showLayer(vector<Node*> layers, int index);

/*functions for Weight*/
MAT* creatMAT(int row, int col);	//create MAT structure of # col input, # row output
void showMAT(MAT* m);	//show the all values of MAT
void initMAT(MAT* m);	//init MAT structure with small random variable divisor determines how it small

/*function for activation*/
double linear(Node* input_layer, int i, MAT* W, int t);		//caculate linear value of output node[i] based on input node and W[output_layer]->mat[i] if t == 1 then transpose
double sigmoid(Node* input_layer, int i, MAT* W, int t);	//caculate sigmoid value of output node[i] based on input node and W[output_layer]->mat[i] if t == 1 then transpose

