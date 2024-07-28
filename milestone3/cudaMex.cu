#include "mex.h"
#include <cuda_runtime.h>
#include <cmath>

#define PI 3.14159265
#define MAX_THETA 180

__global__ void houghTransformKernel(unsigned char* d_image, int width, int height, int* d_accumulator, int max_rho, int num_thetas) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        if (d_image[y * width + x] > 0) {
            for (int theta = 0; theta < num_thetas; ++theta) {
                float theta_rad = (theta * PI) / 180.0;
                int rho = (int)(x * cos(theta_rad) + y * sin(theta_rad));
                int rho_index = rho + max_rho;  // Shift rho index to positive

                if (rho_index >= 0 && rho_index < 2 * max_rho + 1) {
                    atomicAdd(&d_accumulator[theta * (2 * max_rho + 1) + rho_index], 1);
                }
            }
        }
    }
}

void houghTransformCUDA(unsigned char* image, int width, int height, int** accumulator, int* max_rho, int* num_thetas) {
    int* d_accumulator;
    unsigned char* d_image;
    int device = 0;

    cudaGetDevice(&device);

    *max_rho = (int)(sqrt(width * width + height * height));
    *num_thetas = MAX_THETA;

    size_t image_size = width * height * sizeof(unsigned char);
    size_t accumulator_size = (*num_thetas) * (2 * (*max_rho) + 1) * sizeof(int);

    *accumulator = (int*)calloc((*num_thetas) * (2 * (*max_rho) + 1), sizeof(int));

    cudaMalloc((void**)&d_image, image_size);
    cudaMalloc((void**)&d_accumulator, accumulator_size);

    cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice);
    cudaMemset(d_accumulator, 0, accumulator_size);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    houghTransformKernel<<<gridSize, blockSize>>>(d_image, width, height, d_accumulator, *max_rho, *num_thetas);

    cudaDeviceSynchronize();

    cudaMemcpy(*accumulator, d_accumulator, accumulator_size, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_accumulator);
}

// MEX gateway function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 3) {
        mexErrMsgIdAndTxt("CUDAHoughTransform:InvalidInput", "Three inputs required (image, width, height).");
    }
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("CUDAHoughTransform:InvalidOutput", "One output required.");
    }

    // Get the input image (assume it's a 1D array of type uint8)
    if (!mxIsUint8(prhs[0])) {
        mexErrMsgIdAndTxt("CUDAHoughTransform:InvalidInputType", "Input must be a uint8 array.");
    }

    unsigned char *image = (unsigned char *)mxGetData(prhs[0]);
    int width = (int)mxGetScalar(prhs[1]);
    int height = (int)mxGetScalar(prhs[2]);

    // Define the output variables
    int max_rho;
    int num_thetas;
    int *accumulator;

    // Call the CUDA function
    houghTransformCUDA(image, width, height, &accumulator, &max_rho, &num_thetas);

    // Create the output array in MATLAB
    mwSize dims[2] = { num_thetas, 2 * max_rho + 1 };
    plhs[0] = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);
    int *output = (int *)mxGetData(plhs[0]);

    // Copy the accumulator to the output
    memcpy(output, accumulator, num_thetas * (2 * max_rho + 1) * sizeof(int));

    // Free the host memory allocated in the CUDA function
    free(accumulator);
}
