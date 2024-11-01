#include <stdlib.h>
#include <stdio.h>

void MatrixMulOnHost(float* M, float* N, float* P, int width) {

    float total, product, value1, value2;

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            total = 0;
            for (int k = 0; k < width; k++) {
                value1 = M[i * width + k];
                value2 = N[k * width + j];
                product = value1 * value2;
                total += product;
            }
            P[i * width + j] = total;
        }
    }
}

int main() {
    int size = 1000;

    // allocate memory for the arrays
    float* x = malloc(sizeof(float) * size * size);
    float* y = malloc(sizeof(float) * size * size);
    float* z = malloc(sizeof(float) * size * size);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            x[i * size + j] = 1; // x[i][j]
            y[i * size + j] = 1;
        }
    }

    MatrixMulOnHost(x, y, z, size);

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

    return 0;
}