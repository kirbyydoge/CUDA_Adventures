#include <stdio.h>
#include "imageutils.cuh"

// weights of each color channel
#define RED_WEIGHT 		(0.299f)
#define GREEN_WEIGHT 	(0.587f)
#define BLUE_WEIGHT		(0.114f)

// dimensions of the thread blocks
#define NUM_BLOCKS_X 	16
#define NUM_BLOCKS_Y	16

__global__
void rgba_to_negative(
	uchar4 *rgbaImage,
	unsigned char*grayscaleImage,
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

	// averaging the values does not work well
	// as our eyes are not receptive to all colors equally
	// these weights give a better feeling grayscale image
	grayscaleImage[idx] =	rgbaImage[idx].x * RED_WEIGHT +
							rgbaImage[idx].y * GREEN_WEIGHT +
							rgbaImage[idx].z * BLUE_WEIGHT;
}

int main() {
// load input picture
	PPMImage *input_image = readPPM("../PPMImages/Poivron.ppm");
	const int dim_x = input_image->x;
	const int dim_y = input_image->y;

	// dimension and size of both arrays for testing
	const int RGB_SIZE = dim_x * dim_y;
	const int RGB_BYTES = RGB_SIZE * sizeof(uchar4);
	const int GRAYSCALE_SIZE = dim_x * dim_y;
	const int GRAYSCALE_BYTES = GRAYSCALE_SIZE * sizeof(unsigned char);

	// calculating grid and block dimensions of threads
	int grid_size_x = (dim_x + NUM_BLOCKS_X - 1) / NUM_BLOCKS_X;
	int grid_size_y = (dim_y + NUM_BLOCKS_Y - 1) / NUM_BLOCKS_Y;
	dim3 grid_dims = dim3(grid_size_x, grid_size_y, 1);
	dim3 block_dims = dim3(NUM_BLOCKS_X, NUM_BLOCKS_Y, 1);

	// memory pointers
	uchar4 *h_image_rgb;
	unsigned char *h_image_grayscale;
	uchar4 *d_image_rgb;
	unsigned char *d_image_grayscale;

	// memory allocation on host
	h_image_rgb = PPM_to_uchar4(input_image, 255);
	h_image_grayscale = (unsigned char *) malloc(GRAYSCALE_BYTES);

	// memory allocation on device
	cudaMalloc((void **) &d_image_rgb, RGB_BYTES);
	cudaMalloc((void **) &d_image_grayscale, GRAYSCALE_BYTES);

	// transferring input array to device memory
	cudaMemcpy(d_image_rgb, h_image_rgb, RGB_BYTES, cudaMemcpyHostToDevice);

	// launching kernels
	rgba_to_negative<<<grid_dims, block_dims>>>(
		d_image_rgb,
		d_image_grayscale,
		dim_x, dim_y
	);

	// getting back the negative image
	cudaMemcpy(h_image_grayscale, d_image_grayscale, GRAYSCALE_BYTES, cudaMemcpyDeviceToHost);

	// save resulting file
	writeGrayScale("../PPMResults/Poivron_gray.pgm", h_image_grayscale, dim_x, dim_y);

	// free host memory
	free(h_image_rgb);
	free(h_image_grayscale);

	// free device memory
	cudaFree(d_image_rgb);
	cudaFree(d_image_grayscale);

	return 0;
}