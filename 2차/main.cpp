#include "structure.h"
#include "RBM.h"
#include "training.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>
#include <vector>
#include <algorithm>

//class name은 c0, c1으로 가정
#define BUF 256
#define CLASS_NUM 2	//number of classes
#define EPOACH 200
#define THRESHOLD 0.5

using namespace std;

DataSet training_set, test_set;	//training and test data set
char buffer;	//for clearing input buffer

void swap() {

}
void shuffle(DataSet data_set) {

}
int main() {
	srand(time(NULL));	//setting seed value

	FILE *trn, *tst, *fp;
	clock_t start, end;

	int N;	//size of training data set
	int Nr[2] = { 0, };	//size of each class

	int layer_num;	//the number of hidden layer
	int matrix_num;
	vector<Node*> layers;	//nodes of each layers
	vector<int> node_num;	//the number of node of each layer
	vector<MAT*> W;	//vector of W matrix (if n layer exists there are n weight MAT)
	double** relation;	//relation matrix for calculating BEP

	int TP = 0;
	int FP = 0;
	int FN = 0;
	int TN = 0;

	double max_threshold = -DBL_MAX;
	double min_threshold = DBL_MAX;
	double differ = DBL_MAX;	//for equal error rate
	double EER_x;
	double EER_y;

	//**************************process 0 -> Set parameters*****************************************//
	printf("Set the number of hidden layer : ");
	scanf("%d", &layer_num);
	scanf("%c", &buffer);
	for (int i = 0; i < layer_num+2; i++) {
		//if i == 0 then set the layer 0 has the number of input dimension nodes
		//if i == layer_num + 1 then set the output layer (l+1) has one output node
		int temp;
		Node* layer;
		if (i == 0) {
			temp = DIMENSION;
		}
		else if (i == layer_num + 1) {
			temp = 1;
		}
		else {
			printf("Set the number of node of hidden layer %d : ", i);
			scanf("%d", &temp);
			scanf("%c", &buffer);
		}
		temp++;	//for x0
		node_num.push_back(temp);	//layer i의 노드 개수 저장
		layer = makeLayer(temp);	//temp개의 node를 갖는 layer i생성
		layers.push_back(layer);	//해당 layer i를 layers[i]에 저장
	}
	//check layer...
	for (int i = 0; i < layers.size(); i++) {
		showLayer(layers, i);
	}

	//**************************process 1 -> read training data*************************************//
	start = clock();
	trn = fopen("./src/trn.txt", "r");
	if (trn != NULL) {
		char line[BUF];
		char* str;
		while (fgets(line, BUF, trn)) {
			Data temp;
			int cnt = 1;

			str = strtok(line, " ");
			temp.x[cnt - 1] = atof(str);
			while (str = strtok(NULL, " ")) {
				cnt++;
				if (cnt <= DIMENSION) {
					temp.x[cnt - 1] = atof(str);
				}
				else {
					temp.r = atoi(str);
				}
			}
			training_set.push_back(temp);
		}
	}
	fclose(trn);
	N = training_set.size();
	end = clock();
	printf("training data reading time : %lf....\n", (double)(end - start) / 1000);

	//**************************process 2 -> Set W[layer], V Matrix(V is eqaul to W[last_layer]*************************************//
	//initialize each W[layer](i,j) with small random number
	start = clock();
	for (int i = 0; i < layer_num+1; i++) {
		//row : num of output nodes  cols : num of input nodes
		int row, col;	
		MAT* temp;

		row = node_num[i + 1];
		col = node_num[i];

		temp = creatMAT(row, col);
		initMAT(temp);
		W.push_back(temp);
		printf("\n\nW[%i]\n\n", i);
		showMAT(temp);
	}

	//memory allocation for each relation Matrix
	matrix_num = W.size();
	relation = (double**)malloc(sizeof(double*)*matrix_num);
	for (int level = 0; level < matrix_num; level++) {
		int cur = (matrix_num - level) - 1;
		int node_num = _msize(layers[cur + 1]) / sizeof(Node);
		relation[cur] = (double*)malloc(sizeof(double)*node_num);
	}

	end = clock();
	printf("Making weight matrix time : %lf....\n", (double)(end - start) / 1000);

	//**************************process 3 -> CD algorithm(pre train) for Weight Matrix initialize****************//
	/*
	start = clock();
	for (int i = 0; i < layer_num - 1; i++) {
		RMB(EPOACH, layers[i], layers[i + 1], W[i]);
	}
	end = clock();
	printf("pre training time : %lf....\n", (double)(end - start) / 1000);
	*/
	
	//*************************process 4 -> Data Traing and Parameter Setup****************************//
	start = clock();
	for (int epoach = 0; epoach < EPOACH; epoach++) {
		clock_t temps, tempe;
		temps = clock();
		for (int i = 0; i < N; i++) {
			training(training_set[i],layers, W, relation, THRESHOLD);	//training...
		}
		tempe = clock();
		printf("%d epoach time : %lf....\n", epoach+1,(double)(tempe - temps) / 1000);
	}

	for (int i = 0; i < W.size(); i++) {
		printf("\n\nW[%i]\n\n", i);
		showMAT(W[i]);
	}
	end = clock();
	printf("training time : %lf....\n", (double)(end - start) / 1000);

	for (int i = 0; i < layers.size(); i++) {
		showLayer(layers, i);
	}

	//*************************process 5 -> Read test data ***************************************************//
	start = clock();
	tst = fopen("./src/tst.txt", "r");
	if (trn != NULL) {
		char line[BUF];
		char* str;
		while (fgets(line, BUF, trn)) {
			Data temp;
			int cnt = 1;

			str = strtok(line, " ");
			temp.x[cnt - 1] = atof(str);
			while (str = strtok(NULL, " ")) {
				cnt++;
				if (cnt <= DIMENSION) {
					temp.x[cnt - 1] = atof(str);
				}
				else {
					temp.r = atoi(str);
				}
			}
			test_set.push_back(temp);
		}
	}
	fclose(tst);
	end = clock();
	printf("test data reading time : %lf....\n", (double)(end - start) / 1000);

	//*************************process 6 -> testing ***************************************************//
	start = clock();
	for (int t = 0; t < test_set.size(); t++) {
		Data data = test_set[t];
		double result = testing(data, layers, W);

		if (result >= THRESHOLD) {
			if (data.r == 1)
				TP++;
			else
				FP++;
		}
		else {
			if (data.r == 0)
				TN++;
			else
				FN++;
		}
		if (result >= max_threshold)
			max_threshold = result;
		if (result < min_threshold)
			min_threshold = result;
	}
	end = clock();
	printf("testing time : %lf....\n", (double)(end - start) / 1000);

	printf("\nNum of Test data : %d\n\n", test_set.size());
	printf("********Confusion Matrix********\n");
	printf("\nTP\t%d\tFP\t%d\n", TP, FP);
	printf("\nFN\t%d\tTN\t%d\n", FN, TN);

	printf("\n********************************\n");
	printf("Threshold Interval : [%lf,%lf]\n", min_threshold, max_threshold);

	//************************process 7 -> testing with treshold***********************************
	start = clock();
	fp = fopen("./result.txt", "w");
	if (fp != NULL) {
		for (double threshold = min_threshold; threshold <= max_threshold; threshold += 0.01) {
			TP = FP = TN = FN = 0;	//초기화
			double sum;
			double FPR, TPR;

			for (int t = 0; t < test_set.size(); t++) {
				Data data = test_set[t];
				double result = testing(data, layers, W);

				if (result >= threshold) {
					if (data.r == 1)
						TP++;
					else
						FP++;
				}
				else {
					if (data.r == 0)
						TN++;
					else
						FN++;
				}
			}
			FPR = (double)FP / (double)(TN + FP);
			TPR = (double)TP / (double)(TP + FN);

			sum = TPR + FPR;
			if (abs(sum - 1.0) < differ) {
				differ = abs(sum - 1.0);
				EER_x = FPR;
				EER_y = TPR;
			}
			fprintf(fp, "%lf\t%lf\t%lf\n", threshold, FPR, TPR);
		}
	}
	fclose(fp);
	fprintf(fp, "EER Point : %lf(FPR), %lf(TPR)\n", EER_x, EER_y);
	end = clock();
	printf("threshold testing time : %lf....\n", (double)(end - start) / 1000);
	return 0;
	return 0;
}