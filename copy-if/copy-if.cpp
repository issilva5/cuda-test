#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <algorithm>

int main(){
	std::chrono::steady_clock::time_point begin, endt;
    	std::chrono::duration< double, std::micro > time_span_us;

	const int N = 100000000;
	int *V, *result;
	V = (int*) malloc(N * sizeof(int));
	result = (int*) malloc(N * sizeof(int));

	for(int i = 0; i < N; i++) scanf("%d", &V[i]);
	
	int* end = result;

	begin = std::chrono::steady_clock::now();

	for(int i = 0; i < N; i++){
	
		if(V[i] > 10){
			*end = V[i];
			end++;
		}

	}

	endt = std::chrono::steady_clock::now();
    	time_span_us = std::chrono::duration_cast< std::chrono::duration<double, std::micro> >(endt - begin);

	for(int *i = result; *i != *end; i++) printf("%d\n", *i);

	printf("DURATION, %.5f\n", time_span_us);
	return 0;

}
