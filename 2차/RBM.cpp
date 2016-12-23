#include "RBM.h"
#include <stdio.h>
#include <stdlib.h>

/*여기 모든 함수 index 제대로 검사할것*/
void RMB(int epoch, Node* input, Node* output, MAT* W) {
	double delta_w = 0;
	for (int i = 0; i < _msize(output) / sizeof(Node); i++) {
		output[i].val = sigmoid(input, i, W, 0);
		double random = ((double)rand() / (double)RAND_MAX);
		output[i].output = (random < output[i].val) ? 1 : 0;
	}
	for (int i = 0; i < _msize(input) / sizeof(Node); i++) {
		input[i].val = sigmoid(output, i, W, 1);
		// 여기서 input은 왜 확률이여 시발??
	}
	for (int i = 0; i < _msize(input) / sizeof(Node); i++) {
		for (int j = 0; j < _msize(output) / sizeof(Node); j++) {

		}
	}
}