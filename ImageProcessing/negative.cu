#include <stdio.h>
#include "imageutils.cuh"

// dimensions of the thread blocks
#define NUM_BLOCKS_X 	16
#define NUM_BLOCKS_Y	16

__global__
void rgba_to_negative(
	uchar4 *rgbaImage,
	uchar4 *negativeImage,
	int numRows, int numCols
) {
	// finding pixel assigned to this thread
	int thread_x = blockDim.x * blockIdx.x + threadIdx.x;
	int thread_y = blockDim.y * blockIdx.y + threadIdx.y;
	int idx = thread_x * numCols + thread_y;

	// thread is out of range (happens when block dimensions don't allign)
	if(thread_x >= numRows || thread_y >= numCols) {
	// if (idx < numRows * numCols) should also work
		return;
	}

	negativeImage[idx].x = 255 - rgbaImage[idx].x;
	negativeImage[idx].y = 255 - rgbaImage[idx].y;
	negativeImage[idx].z = 255 - rgbaImage[idx].z;
}

int main() {
	// load input picture
	PPMImage *input_image = readPPM("../PPMImages/Poivron.ppm");
	const int dim_x = input_image->x;
	const int dim_y = input_image->y;

	// dimension and size of both arrays for testing
	const int RGB_SIZE = dim_x * dim_y;
	const int RGB_BYTES = RGB_SIZE * sizeof(uchar4);
	const int NEGATIVE_SIZE = dim_x * dim_y;
	const int NEGATIVE_BYTES = NEGATIVE_SIZE * sizeof(uchar4);

	// calculating grid and block dimensions of threads
	int grid_size_x = (dim_x + NUM_BLOCKS_X - 1) / NUM_BLOCKS_X;
	int grid_size_y = (dim_y + NUM_BLOCKS_Y - 1) / NUM_BLOCKS_Y;
	dim3 grid_dims = dim3(grid_size_x, grid_size_y, 1);
	dim3 block_dims = dim3(NUM_BLOCKS_X, NUM_BLOCKS_Y, 1);

	// memory pointers
	uchar4 *h_image_rgb;
	uchar4 *h_image_negative;
	uchar4 *d_image_rgb;
	uchar4 *d_image_negative;

	// memory allocation on host
	h_image_rgb = PPM_to_uchar4(input_image, 255);
	h_image_negative = (uchar4 *) malloc(NEGATIVE_BYTES);

	// memory allocation on device
	cudaMalloc((void **) &d_image_rgb, RGB_BYTES);
	cudaMalloc((void **) &d_image_negative, NEGATIVE_BYTES);

	// transferring input array to device memory
	cudaMemcpy(d_image_rgb, h_image_rgb, RGB_BYTES, cudaMemcpyHostToDevice);

	// launching kernels
	rgba_to_negative<<<grid_dims, block_dims>>>(
		d_image_rgb,
		d_image_negative,
		dim_x, dim_y
	);

	// getting back the negative image
	cudaMemcpy(h_image_negative, d_image_negative, NEGATIVE_BYTES, cudaMemcpyDeviceToHost);

	// save resulting file
	PPMImage *result = uchar4_to_PPM(h_image_negative, dim_x, dim_y);
	writePPM("../PPMResults/Poivron_neg.ppm", result);

	// free host memory
	free(h_image_rgb);
	free(h_image_negative);

	// free device memory
	cudaFree(d_image_rgb);
	cudaFree(d_image_negative);

	return 0;
}