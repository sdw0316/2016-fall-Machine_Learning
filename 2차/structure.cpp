#include "structure.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

Node* makeLayer(int n) {
	Node* newlayer = (Node*)malloc(sizeof(Node)*n);
	for (int i = 1; i < n; i++) {
		newlayer[i].val = 0.0;
		newlayer[i].output = 0;
	}
	newlayer[0].val = 1.0;
	newlayer[0].output = 0;
	return newlayer;
}

void showLayer(vector<Node*> layers, int index) {
	printf("\nsizeof layer %d : %d\n", index, (_msize(layers[index]) / sizeof(Node)));
	printf("layer %d's nodes : ", index);
	for (int j = 0; j < _msize(layers[index]) / sizeof(Node); j++) {
		printf("%lf ", layers[index][j].val);
	}
	printf("\n");
}

MAT* creatMAT(int row, int col) {
	MAT *m = (MAT*)malloc(sizeof(MAT));
	m->row = row;	
	m->col = col;

	//init weight matrix
	m->matrix = (double**)malloc(sizeof(double*)*(m->row));
	for (int i = 0; i < m->row; i++) {
		m->matrix[i] = (double*)malloc(sizeof(double) * (m->col));
	}
	return m;
}

void showMAT(MAT* m) {
	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) {
			printf("%lf ", m->matrix[i][j]);
		}
		printf("\n");
	}
}

//init Weight matrix with Small Random values divisor deter
void initMAT(MAT* m) {
	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) {
			double value = (double)rand() / (double)RAND_MAX;
			if (rand() % 2 == 0) {
				value *= -1.0;
			}
			m->matrix[i][j] = value / 100.0;
		}
	}
}

double linear(Node* input_layer, int i, MAT* W, int t) {
	double o = 0.0;
	if (!t) {
		for (int j = 0; j < _msize(input_layer) / sizeof(Node); j++) {
			o += (W->matrix[i][j] * input_layer[j].val);
		}
	}
	else {
		for (int j = 0; j < _msize(input_layer) / sizeof(Node); j++) {
			o += (W->matrix[j][i] * input_layer[j].val);
		}
	}
	return o;
}

double sigmoid(Node* input_layer, int i, MAT* W, int t) {
	double o = 0.0;
	if (!t) {
		for (int j = 0; j < _msize(input_layer) / sizeof(Node); j++) {
			o -= (W->matrix[i][j] * input_layer[j].val);
		}
	}
	else {
		for (int j = 0; j < _msize(input_layer) / sizeof(Node); j++) {
			o -= (W->matrix[j][i] * input_layer[j].val);
		}
	}
	return ((1.0) / (1.0 + exp(o)));
}