#include "training.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define RATE 0.00001

int t = 0;
void setData(Data data, Node* layer0) {
	layer0[0].val = 1.0;
	for (int i = 1; i <= DIMENSION; i++) {
		layer0[i].val = data.x[i - 1];
	}
}

void training(Data input, vector<Node*> layers, vector<MAT*> W, double** relation, double threshold) {
	char r = input.r;
	char y;
	double o; //classification value

	int num_layer = layers.size();	//number of hidden layer
	int num_MAT = W.size();

	setData(input, layers[0]);	//set the layer[0] to input data
	//feed forward
	for (int i = 1; i < num_layer; i++) {
		//calculate each hidden layer value
		if(i != num_layer - 1){
			//calculate each node of layer with sigmoid function
			for (int j = 1; j < _msize(layers[i]) / sizeof(Node); j++) {
				layers[i][j].val = sigmoid(layers[i - 1], j, W[i - 1], 0);
			}
		}
		else {
			//calculate output value with leaner function
			for (int j = 1; j < _msize(layers[i]) / sizeof(Node); j++) {
				o = linear(layers[i - 1], 1, W[i - 1], 0);
				layers[i][j].val = o;
				if (o >= threshold) {
					y = 1;
				}
				else {
					y = 0;
				}
			}
		}
	}
	//calculate realtion matrix for back propagation
	for (int level = 1; level < num_layer; level++) {
		int cur = (num_layer) - level;
		for (int in = 0; in < _msize(layers[cur]) / sizeof(Node); in++) {
			relation[cur - 1][in] = 0.0;
			if (level == 1) {
				//for top weight matrix error
				relation[cur - 1][0] = 0.0;
				relation[cur - 1][1] = (double)r - (double)layers[cur][1].val;
			}
			else {
				int out_num = _msize(layers[cur + 1]) / sizeof(Node);	//the number of upper layer's node
				for (int out = 1; out < out_num; out++) {
					relation[cur-1][in] += (relation[cur][out] * W[cur]->matrix[out][in]);
				}
				relation[cur - 1][in] = relation[cur - 1][in] * layers[cur][in].val * (1 - layers[cur][in].val);
				//printf("%lf %d :: \n", relation[num_layer - 2][1], level);
			}
			//printf("%lf \n", relation[num_layer - 2][1]);
			//printf("layer %d %d : %lf\n", cur, in,layers[cur - 1][in].val);
		}
	}
	//adjust weight matrix by using relation matrix
	for (int level = 1; level < num_layer; level++) {
		int cur = (num_layer) - level;
		for (int out = 0; out <_msize(layers[cur]) / sizeof(Node); out++) {
			for (int in = 0; in < _msize(layers[cur - 1]) / sizeof(Node); in++) {
				double delta = relation[cur - 1][out] * layers[cur - 1][in].val;
				W[cur - 1]->matrix[out][in] += (RATE * delta);
			}
		}
	}
}

double testing(Data input, vector<Node*> layers, vector<MAT*> W) {
	double o; //classification value
	int num_layer = layers.size();	//number of hidden layer

	setData(input, layers[0]);	//set the layer[0] to input data
	//feed forward
	for (int i = 1; i < num_layer; i++) {
		//calculate each hidden layer value
		if (i != num_layer - 1) {
			//calculate each node of layer with sigmoid function
			for (int j = 1; j < _msize(layers[i]) / sizeof(Node); j++) {
				layers[i][j].val = sigmoid(layers[i - 1], j, W[i - 1], 0);
			}
		}
		else {
			//calculate output value with leaner function
			o = linear(layers[i - 1], 1, W[i - 1], 0);
		}
	}

	return o;
}