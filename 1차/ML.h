#include <vector>

#define DIMENSION 13	//input vector dimension
#define CLASS_NUM 2	//number of classes

using namespace std;

typedef struct _data {
	double x[13];	//input features
	int c;	//class 0 or 1
}Data;

typedef vector<Data> DataSet;	//data set
