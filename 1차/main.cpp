#include "ML.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>
#include <vector>

//class name은 c0, c1으로 가정

#define BUF 256
#define DIMENSION 13	//input vector dimension
#define CLASS_NUM 2	//number of classes

DataSet training_set, test_set;	//training and test data set

double normal(double x, double mean, double var) {
	return 1 / sqrt(2 * M_PI* var) * exp((float(-1) * (x - mean) * (x - mean) / (2 * var)));
}

int main() {
	FILE *trn, *tst, *fp;
	clock_t start, end;

	int N;	//size of training set
	int number[CLASS_NUM] = { 0, };	//N0, N1
	double p[CLASS_NUM];	//P(C0), P(C1)
	double mean[CLASS_NUM][DIMENSION] = { 0, };	//M(0,1)~ M(0,13), M(1,1)~ M(1,13)
	double var[CLASS_NUM][DIMENSION] = { 0, };	//var(0,1)~var(0,13), var(1,1)~var(1,13)

	double MAX = -DBL_MAX;
	double MIN = DBL_MAX;

	int TP = 0;
	int FP = 0;
	int FN = 0;
	int TN = 0;

	double differ = DBL_MAX;	//for equal error rate
	double EER_x;
	double EER_y;

	//************************process 1 -> read training data************************************
	start = clock();
	trn = fopen("./src/trn.txt","r");
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
					temp.c = atoi(str);
				}
			}
			training_set.push_back(temp);
		}
	}
	fclose(trn);
	N = training_set.size();
	end = clock();
	printf("training data reading time : %lf....\n", (double)(end - start) / 1000);

	//**********************process 2 -> learning***********************************************
	start = clock();
	for (int k = 0; k < CLASS_NUM; k++) {
		//store the number of each class 
		for (int t = 0; t < N; t++) {
			if (training_set[t].c == k)
				number[k]++;
		}
		p[k] = ((double)number[k] / N); //store the prior of each class
		
		for (int i = 0; i < DIMENSION; i++) {
			//calculate the sample means
			for (int t = 0; t < N; t++) {
				if (training_set[t].c == k) {
					mean[k][i] += training_set[t].x[i];
				}
			}
			mean[k][i] /= number[k];
			//calculate the sample variences
			for (int t = 0; t < N; t++) {
				if (training_set[t].c == k) {
					var[k][i] += ((training_set[t].x[i]-mean[k][i])*(training_set[t].x[i] - mean[k][i]));
				}
			}
			var[k][i] /= number[k];
		}
	}
	end = clock();
	printf("learning time : %lf....\n", (double)(end - start) / 1000);

	//**********************process 3 -> lead test data***********************************************
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
					temp.c = atoi(str);
				}
			}
			test_set.push_back(temp);
		}
	}
	fclose(tst);
	end = clock();
	printf("test data reading time : %lf....\n", (double)(end - start) / 1000);

	int test_t = 0;
	int test_f = 0;
	//************************process 4 -> testing************************************
	start = clock();
	for (int t = 0; t < test_set.size(); t++) {
		Data temp = test_set[t];
		int result = 0;
		double g[CLASS_NUM] = { 0, };	//classfication probability
		double threshold;

		for (int k = 0; k < CLASS_NUM; k++) {
			for (int i = 0; i < DIMENSION; i++) {
				g[k] += (log(var[k][i]) + (((temp.x[i] - mean[k][i])*(temp.x[i] - mean[k][i]) / (var[k][i]))));
			}
			g[k] -= log(p[k]*p[k]);
			g[k] *= (-1.0);
		}

		threshold = g[1] - g[0];
		if (g[1] > g[0]) {
			result = 1;
			if (threshold > MAX)
				MAX = threshold;
		}
		else {
			if (threshold < MIN)
				MIN = threshold;
		}

		//calculate confusion MAT
		if (result == 1) {
			test_t++;
			if (result == temp.c)
				TP++;
			else
				FP++;
		}
		else {
			test_f++;
			if (result == temp.c)
				TN++;
			else
				FN++;
		}
	}
	end = clock();
	printf("testing time : %lf....\n", (double)(end - start) / 1000);
	
	printf("\nNum of Test data : %d\n\n",test_set.size());
	printf("********Confusion Matrix********\n");
	printf("\nTP\t%d\tFP\t%d\n",TP,FP); 
	printf("\nFN\t%d\tTN\t%d\n", FN, TN);

	printf("\n********************************\n");
	printf("Threshold Interval : [%lf,%lf]\n", -MAX, -MIN);

	//************************process 5 -> testing with treshold***********************************
	start = clock();
	fp = fopen("./result.txt", "w");
	if (fp != NULL) {
		for (double threshold = -MAX; threshold <= -MIN; threshold += 0.1) {
			TP = FP = TN = FN = 0;	//초기화
			double sum;
			double FPR, TPR;

			for (int t = 0; t < test_set.size(); t++) {
				Data temp = test_set[t];
				int result = 0;
				double g[CLASS_NUM] = { 0, };	//classfication probability

				for (int k = 0; k < CLASS_NUM; k++) {
					for (int i = 0; i < DIMENSION; i++) {
						g[k] += (log(var[k][i]) + (((temp.x[i] - mean[k][i])*(temp.x[i] - mean[k][i]) / (var[k][i]))));
					}
					g[k] -= log(p[k] * p[k]);
					g[k] *= (-1.0);
				}

				if (g[1] + threshold > g[0]) {
					result = 1;
				}

				//calculate confusion MAT
				if (result == 1) {
					test_t++;
					if (result == temp.c)
						TP++;
					else
						FP++;
				}
				else {
					test_f++;
					if (result == temp.c)
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
			fprintf(fp,"%lf\t%lf\t%lf\n", threshold, FPR, TPR);
		}
	}
	fclose(fp);
	printf("EER Point : %lf(FPR), %lf(TPR)\n", EER_x, EER_y);
	end = clock();
	printf("threshold testing time : %lf....\n", (double)(end - start) / 1000);
	return 0;
}