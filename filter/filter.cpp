#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <algorithm>

int main(){
	std::chrono::steady_clock::time_point begin, endt;
    	std::chrono::duration< double, std::micro > time_span_us;

	const int N = 7000;
	int *V, *result;
	V = (int*) malloc(N * sizeof(int));
	result = (int*) malloc(N * sizeof(int));
	int valid = 0;

	for(int i = 0; i < N; i++) scanf("%d", &V[i]);
	
	int* end = result;

	begin = std::chrono::steady_clock::now();

	for(int i = 0; i < N; i++){
	
		if(V[i] > 0){
			*end = V[i];
			end++;
			valid++;
		}

	}

	endt = std::chrono::steady_clock::now();
    	time_span_us = std::chrono::duration_cast< std::chrono::duration<double, std::micro> >(endt - begin);

	for(int *i = result; *i != *end; i++) printf("%d\n", *i);

	printf("DURATION, %.5f\n", time_span_us);
	printf("VALID, %d\n", valid);
	printf("VALID FRACTION, %.5f\n", valid*1.0/N);
	return 0;

}
