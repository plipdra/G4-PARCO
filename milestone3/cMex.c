#include "mex.h"
#include <math.h>
#include <stdlib.h>

#define PI 3.14159265
#define MAX_THETA 180

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

void houghTransform(unsigned char* image, int width, int height, int** accumulator, int* max_rho, int* num_thetas) {
    int x, y, theta, theta_max, theta_min;
    int rho;
    double theta_rad;

    theta_min = -90;
    theta_max = 89;

    *max_rho = (int)(sqrt(width * width + height * height));
    *num_thetas = theta_max - theta_min + 1;

    *accumulator = (int*)calloc((*num_thetas) * (2 * (*max_rho) + 1), sizeof(int));

    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            if (image[y + x * height] > 0) {  // Correct index for column-major order
                for (theta = theta_min; theta <= theta_max; ++theta) {
                    theta_rad = (theta * PI) / 180.0;
                    rho = (int)(x * cos(theta_rad) + y * sin(theta_rad));
                    int theta_index = theta - theta_min;
                    int rho_index = rho + *max_rho;  // Shift rho index to positive

                    if (rho_index >= 0 && rho_index < 2 * (*max_rho) + 1) {
                        (*accumulator)[theta_index * (2 * (*max_rho) + 1) + rho_index]++;
                    }
                }
            }
        }
    }
}

void saveAccumulatorAsText(int* accumulator, int max_rho, int num_thetas, const char* filename) {
    int total_rho = 2 * max_rho;

    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Failed to open file for writing\n");
        return;
    }

    for (int rho = 0; rho < total_rho; rho++) {
        for (int theta = 0; theta < num_thetas; theta++) {
            fprintf(file, "%d ", accumulator[theta * total_rho + rho]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs != 3) {
        mexErrMsgIdAndTxt("HoughTransform:invalidNumInputs", "Three inputs required.");
    }
    if (nlhs != 3) {
        mexErrMsgIdAndTxt("HoughTransform:invalidNumOutputs", "Three outputs required.");
    }

    unsigned char* image = (unsigned char*)mxGetData(prhs[0]);
    int width = (int)mxGetScalar(prhs[1]);
    int height = (int)mxGetScalar(prhs[2]);

    int* accumulator;
    int max_rho, num_thetas;

    houghTransform(image, width, height, &accumulator, &max_rho, &num_thetas);
    saveAccumulatorAsPGM(accumulator, max_rho, num_thetas, "accumulator1024C.pgm");
    saveAccumulatorAsText(accumulator, max_rho, num_thetas, "accumulator.txt");

    mwSize dims[2] = { (mwSize)num_thetas, (mwSize)(2 * max_rho + 1) };
    plhs[0] = mxCreateNumericArray(2, dims, mxUINT8_CLASS, mxREAL);
    unsigned char* outData = (unsigned char*)mxGetData(plhs[0]);

    int max_value = 0;
    for (int i = 0; i < num_thetas * (2 * max_rho + 1); ++i) {
        if (accumulator[i] > max_value) {
            max_value = accumulator[i];
        }
    }
    printf("highest: %d \n", max_value);

    for (int theta = 0; theta < num_thetas; ++theta) {
        for (int rho = 0; rho < 2 * max_rho + 1; ++rho) {
            int value = 255 * accumulator[theta * (2 * max_rho + 1) + rho];
            outData[theta + rho * num_thetas] = (unsigned char)(value / max_value);
        }
    }

    plhs[1] = mxCreateDoubleScalar((double)num_thetas);
    plhs[2] = mxCreateDoubleScalar((double)max_rho);

    free(accumulator);
}
