#include <stdlib.h>
#include <stdio.h>

#define TILE_WIDTH 16 // 16 * 16 = 256

__global__ void MatMul(float* d_M, float* d_N, float* d_P, int Width){
    int Row = blockIdx.y*blockDim.y+threadIdx.y;
    int Col = blockIdx.x*blockDim.x+threadIdx.x;

    if ((Row < Width) && (Col < Width)){
        float value1, value2;
        float Pvalue = 0;
        for (int k = 0; k < Width; k++){
            value1 = d_M[Row * Width + k];
            value2 = d_N[k * Width + Col];
            Pvalue += value1 * value2;
            d_P[Row * Width + Col] = Pvalue;
        }
    }
}

int main(){
    int size = 1000;

    // allocate memory for the arrays
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

    // calculate a "2D" thread and block call
    dim3 dimGrid(ceil((1.0*size)/TILE_WIDTH), ceil((1.0*size)/TILE_WIDTH), 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    MatMul<<<dimGrid, dimBlock>>>(x, y, z, size);
    cudaDeviceSynchronize();

    // // print the output for small size testing
    // for (int i = 0; i < size; i++){
    //     for (int j = 0; j < size; j++){
    //         printf("[%f] ", z[i * size + j]);
    //     }
    //     printf("\n");
    // }

    // check for errors (presumes array is only 1's)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (z[i * size + j] != size) {
                printf("Error at z[%d][%d]: %f\n", i, j, z[i * size + j]);
            }
        }
    }

    // free memory
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    return 0;
}