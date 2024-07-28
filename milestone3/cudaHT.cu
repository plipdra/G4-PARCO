#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <windows.h>
#include <math.h>
#include <cuda_runtime.h>

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

    cudaMallocManaged((void**)&d_image, image_size);
    cudaMallocManaged((void**)&d_accumulator, accumulator_size);

    memcpy(d_image, image, image_size);

    cudaMemPrefetchAsync(d_image, image_size, device);
    cudaMemPrefetchAsync(d_accumulator, accumulator_size, device);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    houghTransformKernel<<<gridSize, blockSize>>>(d_image, width, height, d_accumulator, *max_rho, *num_thetas);

    cudaDeviceSynchronize();

    cudaMemPrefetchAsync(d_accumulator, accumulator_size, cudaCpuDeviceId);

    memcpy(*accumulator, d_accumulator, accumulator_size);

    cudaFree(d_image);
    cudaFree(d_accumulator);
}
void houghTransform(unsigned char* image, int width, int height, int** accumulator, int* max_rho, int* num_thetas) {
    int x, y, theta;
    int rho;
    float theta_rad;

    *max_rho = (int)(sqrt(width * width + height * height));
    *num_thetas = MAX_THETA;

    *accumulator = (int*)calloc((*num_thetas) * (2 * (*max_rho) + 1), sizeof(int));

    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            if (image[y * width + x] > 0) {
                for (theta = 0; theta < *num_thetas; ++theta) {
                    theta_rad = (theta * PI) / 180.0;
                    rho = (int)(x * cos(theta_rad) + y * sin(theta_rad));
                    int rho_index = rho + *max_rho;  // Shift rho index to positive

                    if (rho_index >= 0 && rho_index < 2 * (*max_rho) + 1) {
                        (*accumulator)[theta * (2 * (*max_rho) + 1) + rho_index]++;
                    }
                }
            }
        }
    }
}
void saveAccumulatorAsPGM(int* accumulator, int max_rho, int num_thetas, const char* filename);

int main() {
    printf("Creating a diagonal line with dimensions 32x32....\n");
    int width = 32;
    int height = 32;

    unsigned char image[1024] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    int* accumulator;
    int max_rho;
    int num_thetas;
    _LARGE_INTEGER freq,start, end;
    double timeTaken, timeTakenC, avgT, avgTC;
    int n = 50;
    timeTaken = 0;
    timeTakenC = 0;
    QueryPerformanceFrequency(&freq);
    printf("Performing C Hough Transform....\n");
    for(int i = 0; i<n; i++){
        QueryPerformanceCounter(&start);
        houghTransform(image, width, height, &accumulator, &max_rho, &num_thetas);
        QueryPerformanceCounter(&end);
        timeTakenC += (double)(end.QuadPart - start.QuadPart) * 1000 / freq.QuadPart;
    }

    avgTC = timeTakenC/n;
    printf("Performing CUDA Hough Transform....\n");
    for(int i = 0; i< n; i++){
        QueryPerformanceCounter(&start);
        houghTransformCUDA(image, width, height, &accumulator, &max_rho, &num_thetas);
        QueryPerformanceCounter(&end);
        timeTaken += (double)(end.QuadPart - start.QuadPart) * 1000 / freq.QuadPart;
    }

    avgT = timeTaken/n;

    printf("Done!\n\nAverage Time Taken to perform Hough Transform with C after %d runs: %fms\nAverage Time Taken to perform Hough Transform with CUDA after %d runs: %fms\n", n,avgTC, n,avgT);
    printf("Saving Accumulator as PGM....\n");
    saveAccumulatorAsPGM(accumulator, max_rho, num_thetas, "accumulator1024_cuda.pgm");

    printf("Saved! Please run IrfanView to see the waveform.\n");

    free(accumulator);
    return 0;
}

void saveAccumulatorAsPGM(int* accumulator, int max_rho, int num_thetas, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file for writing\n");
        return;
    }

    int width = 2 * max_rho + 1;
    int height = num_thetas;

    int max_value = 0;
    for (int i = 0; i < width * height; ++i) {
        if (accumulator[i] > max_value) {
            max_value = accumulator[i];
        }
    }

    fprintf(file, "P5\n%d %d\n255\n", width, height);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            unsigned char pixel = (unsigned char)(255.0 * accumulator[i * width + j] / max_value);
            fwrite(&pixel, sizeof(unsigned char), 1, file);
        }
    }

    fclose(file);
}

