#include <hip/hip_runtime.h>
#include <stdio.h>
#include <math.h>

#define DEBUG 0

// TODO: add a device kernel that copies all elements of a vector
//       using GPU threads in a 2D grid
#define HIP_SAFECALL(x) {      \
  hipError_t status = x;       \
  if (status != hipSuccess) {  \
    printf("HIP Error: %s\n", hipGetErrorString(status));  \
  } }

// from A to B
__global__ void dcopy_(int nr, int nc, double *A, double *B)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int limx = (x+blockDim.y)<nc?(x+blockDim.y):nc;
    if (DEBUG && x==0 && y==0) printf("blockDim.x=%i, blockDim.y=%i\n", blockDim.x, blockDim.y);
    if (DEBUG) printf("Block (%3.i,%3.i), Thread (%3.i,%3.i), solving for inds=%3.i-%3.i\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, x, limx);
    if (y < nr) {
        for (; x < limx; ++x ) {
            int ind = y*nc + x;
            B[ind] = A[ind];
        }
    }

}



int main(void)
{
    int i, j;
    const int n = 600;
    const int m = 400;
    const int size = n * m;
    double x[size], y[size], y_ref[size];
    double *x_, *y_;

    // initialise data
    for (i=0; i < size; i++) {
        x[i] = (double) i / 1000.0;
        y[i] = 0.0;
    }
    // copy reference values (C ordered)
    for (i=0; i < n; i++) {
        for (j=0; j < m; j++) {
            y_ref[i * m + j] = x[i * m + j];
        }
    }

    // TODO: allocate vectors x_ and y_ on the GPU
    // TODO: copy initial values from CPU to GPU (x -> x_ and y -> y_)
    HIP_SAFECALL(hipMalloc(&x_, sizeof(double) * size));
    HIP_SAFECALL(hipMalloc(&y_, sizeof(double) * size));
    HIP_SAFECALL(hipMemcpy(x_, x, sizeof(double) * size, hipMemcpyHostToDevice));
    HIP_SAFECALL(hipMemcpy(y_, y, sizeof(double) * size, hipMemcpyHostToDevice));



    // TODO: define grid dimensions (use 2D grid!)
    // 13*32=416
    // assume every thread makes 300 steps -> 2 block-cols
    dim3 blocks(13, 2, 1);
    dim3 threads(1, 32, 1);
    
    // TODO: launch the device kernel
    hipLaunchKernelGGL(dcopy_, blocks, threads, 0, 0, n, m, x_, y_);
    HIP_SAFECALL(hipGetLastError()); 


    // TODO: copy results back to CPU (y_ -> y)
    HIP_SAFECALL(hipMemcpy(y, y_, sizeof(double) * size, hipMemcpyDeviceToHost));

    // confirm that results are correct
    double error = 0.0;
    for (i=0; i < size; i++) {
        error += abs(y_ref[i] - y[i]);
    }
    printf("total error: %f\n", error);
    printf("  reference: %f at (42,42)\n", y_ref[42 * m + 42]);
    printf("     result: %f at (42,42)\n", y[42 * m + 42]);

    return 0;
}
