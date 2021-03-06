// Following from the course of "Udacity - Intro to Parallel Programming"
// Simple program to square each element of a given vector in parallel

#include <stdio.h>

#define ARRAY_SIZE 64
#define ARRAY_BYTES (ARRAY_SIZE * sizeof(float))

__global__
void square(float *d_out, float *d_in) {
	int idx = threadIdx.x;
	float f = d_in[idx];
	d_out[idx] = f * f;
}

int main(int argc, char *argv[]) {
	// generate the input array on the host
	float h_in[ARRAY_SIZE];
	for(int i = 0; i < ARRAY_SIZE; i++) {
		h_in[i] = float(i);
	}
	float h_out[ARRAY_SIZE];

	// declare GPU memory pointers
	float *d_in;
	float *d_out;

	// allocate GPU mmemory
	cudaMalloc((void **) &d_in, ARRAY_BYTES);
	cudaMalloc((void **) &d_out, ARRAY_BYTES);

	// transfer the array to the GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	// launch the kernel
	square<<<1, ARRAY_SIZE>>>(d_out, d_in);

	// cope back the result array to the GPU
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	// print out the resulting array
	for(int i = 0; i < ARRAY_SIZE; i++) {
		printf("%10.2f", h_out[i]);
		printf(((i % 4) != 3) ? "\t" : "\n");
	} 

	// free GPU memory allocation
	cudaFree(d_in);
	cudaFree(d_out);

	return 0;
}