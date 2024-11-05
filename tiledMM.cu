#include <stdlib.h>
#include <stdio.h>

#define TILE_WIDTH 16 // 16 * 16 = 256

__global__ void MatMul(float* d_M, float* d_N, float* d_P, int Width){
    __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = blockIdx.y*TILE_WIDTH+ty;
    int Col = blockIdx.x*TILE_WIDTH+tx;
    float Pvalue = 0;

    for (int m = 0; m < Width/TILE_WIDTH; m++){
        subTileM[ty][tx] = d_M[Row * Width + m * TILE_WIDTH + tx];
        subTileN[ty][tx] = d_N[(m * TILE_WIDTH + ty) * Width + Col];
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++){
            float value1, value2;
            value1 = subTileM[ty][k];
            value2 = subTileN[k][tx];
            Pvalue += value1 * value2;
            d_P[Row * Width + Col] = + Pvalue;
        }
        __syncthreads();
    }
}

int main(){
    int size = 128; // test 128, 256, 512

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