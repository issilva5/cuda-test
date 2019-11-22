#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <stdio.h>
#include <stdlib.h>
//#include <thrust/sort.h>
#include <chrono>

struct is_even
{
  __host__ __device__
  bool operator()(const int x)
  {
    return (x > 10);
  }
};

int main(){

	std::chrono::steady_clock::time_point begin, endt;
    	std::chrono::duration< double, std::micro > time_span_us;	

	const int N = 100000000;
	int *V, *result;
	V = (int*) malloc(N * sizeof(int));
	result = (int*) malloc(N * sizeof(int));

	for(int i = 0; i < N; i++) scanf("%d", &V[i]);

	begin = std::chrono::steady_clock::now();
	int *end = thrust::copy_if(V, V + N, result, is_even());
	endt = std::chrono::steady_clock::now();
    	time_span_us = std::chrono::duration_cast< std::chrono::duration<double, std::micro> >(endt - begin);

	for(int *i = result; *i != *end; i++) printf("%d\n", *i);
	printf("DURATION, %.5f\n", time_span_us);
	return 0;

}
