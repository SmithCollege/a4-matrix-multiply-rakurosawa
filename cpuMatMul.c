#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

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

double get_clock() {
    struct timeval tv; int ok;
    ok = gettimeofday(&tv, (void *) 0);
    if (ok<0) { 
        printf("gettimeofday error"); 
        }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}


int main() {
    int size = 512;

    // allocate memory for the arrays
    float* x = malloc(sizeof(float) * size * size);
    float* y = malloc(sizeof(float) * size * size);
    float* z = malloc(sizeof(float) * size * size);

    double t0, t1;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            x[i * size + j] = 1; // x[i][j]
            y[i * size + j] = 1;
        }
    }

    t0 = get_clock();
    MatrixMulOnHost(x, y, z, size);
    t1 = get_clock();

    printf("time: %f ns\n", 1000000000.0*(t1-t0));

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