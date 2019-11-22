#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda_runtime.h>
#include "device_functions.h"
#include <cuda_profiler_api.h>

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


__global__ void filter_k(int *dst, int *nres, const int *src, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  while(i < n){

    if(src[i] > 0) dst[atomicAdd(nres, 1)] = src[i];
    i += blockDim.x * gridDim.x;

  }

}

int main(){

	std::chrono::steady_clock::time_point begin, endt, beginop, endop;
    	std::chrono::duration< double, std::micro > time_span_us, time_span_us_op;	

	const int N = 7000;
	int *V, *result;
	int valid = 0;
	V = (int*) malloc(N * sizeof(int));
	result = (int*) malloc(N * sizeof(int));

	for(int i = 0; i < N; i++) scanf("%d", &V[i]);

	begin = std::chrono::steady_clock::now();

	int *dst, *nres, *src;
	HANDLE_ERROR(cudaMalloc((void** ) &dst, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void** ) &nres, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void** ) &src, N * sizeof(int)));

	HANDLE_ERROR(cudaMemcpy(dst, result, N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(src, V, N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(nres, &valid, sizeof(int), cudaMemcpyHostToDevice));
	
	beginop = std::chrono::steady_clock::now();

	filter_k<<<1024, 512>>>(dst, nres, src, N);

	endop = std::chrono::steady_clock::now();
	
	HANDLE_ERROR(cudaMemcpy(&valid, nres, sizeof(int), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(result, dst, N * sizeof(int), cudaMemcpyDeviceToHost));
	
	cudaFree(dst);
	cudaFree(nres);
	cudaFree(src);

	endt = std::chrono::steady_clock::now();
    	time_span_us = std::chrono::duration_cast< std::chrono::duration<double, std::micro> >(endt - begin);
	
	time_span_us_op = std::chrono::duration_cast< std::chrono::duration<double, std::micro> >(endop - beginop);	

	for(int i = 0; i < valid; i++) printf("%d\n", result[i]);
	printf("DURATION OP, %.5f\n", time_span_us_op);
	printf("DURATION, %.5f\n", time_span_us);
	printf("VALID, %d\n", valid);
	printf("VALID FRACTION, %.5f\n", valid*1.0/N);
	return 0;

}
