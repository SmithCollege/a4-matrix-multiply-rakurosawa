#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>

// compile as: nvcc cublasMM.cu -o cublasMM -lcublas

int main(){
    int size = 1000;

    float *x, *y, *z;
    cudaMallocManaged(&x, sizeof(float) * size * size);
    cudaMallocManaged(&y, sizeof(float) * size * size);
    cudaMallocManaged(&z, sizeof(float) * size * size);

    // initialize values for x and y arrays
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            x[i * size + j] = 1; // x[i][j]
            y[i * size + j] = 1;
        }
    }
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha, x, size, y, size, &beta, z, size);

    cublasDestroy(handle);

    cudaDeviceSynchronize();

        for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (z[i * size + j] != size) {
                printf("Error at z[%d][%d]: %f\n", i, j, z[i * size + j]);
            }
        }
    }

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    return 0;
}