#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <sys/time.h>

// compile as: nvcc cublasMM.cu -o cublasMM -lcublas

double get_clock() {
    struct timeval tv; int ok;
    ok = gettimeofday(&tv, (void *) 0);
    if (ok<0) { 
        printf("gettimeofday error"); 
        }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

int main(){
    int size = 1024;

    float *x, *y, *z;
    cudaMallocManaged(&x, sizeof(float) * size * size);
    cudaMallocManaged(&y, sizeof(float) * size * size);
    cudaMallocManaged(&z, sizeof(float) * size * size);

    double t0, t1;

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

    t0 = get_clock();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha, x, size, y, size, &beta, z, size);
    cublasDestroy(handle);
    cudaDeviceSynchronize();
    t1 = get_clock();

    printf("time: %f ns\n", 1000000000.0*(t1-t0));


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