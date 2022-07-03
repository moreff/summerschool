#include <hip/hip_runtime.h>
#include <stdio.h>
#include <math.h>

#define HIP_SAFECALL(x) {      \
  hipError_t status = x;       \
  if (status != hipSuccess) {  \
    printf("HIP Error: %s\n", hipGetErrorString(status));  \
  } }

// TODO: add a device kernel that calculates y = a * x + y
__global__ void kernel_(int n, float a, float *x, float *y)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (; tid < n; tid += stride) {
        y[tid] = a*x[tid] + y[tid];
    }
}

int main(void)
{
    int i;
    const int n = 10000;
    float a = 3.4;
    float x[n], y[n], y_ref[n];
    float *x_, *y_;

    // initialise data and calculate reference values on CPU
    for (i=0; i < n; i++) {
        x[i] = sin(i) * 2.3;
        y[i] = cos(i) * 1.1;
        y_ref[i] = a * x[i] + y[i];
    }

    // TODO: allocate vectors x_ and y_ on the GPU
    HIP_SAFECALL(hipMalloc(&x_, sizeof(float) * n));
    HIP_SAFECALL(hipMalloc(&y_, sizeof(float) * n));

    // TODO: copy initial values from CPU to GPU (x -> x_ and y -> y_)
    HIP_SAFECALL(hipMemcpy(x_, x, sizeof(float) * n, hipMemcpyHostToDevice));
    HIP_SAFECALL(hipMemcpy(y_, y, sizeof(float) * n, hipMemcpyHostToDevice));


    // TODO: define grid dimensions
    dim3 blocks(32);
    dim3 threads(256);
    // TODO: launch the device kernel
    //HIP_SAFECALL(hipLaunchKernelGGL(kernel_, blocks, threads, 0, 0, n, a, x_, y_));
    hipLaunchKernelGGL(kernel_, blocks, threads, 0, 0, n, a, x_, y_);
    HIP_SAFECALL(hipGetLastError()); 

    // TODO: copy results back to CPU (y_ -> y)
    HIP_SAFECALL(hipMemcpy(y, y_, sizeof(float) * n, hipMemcpyDeviceToHost));


    // confirm that results are correct
    float error = 0.0;
    float tolerance = 1e-6;
    float diff;
    for (i=0; i < n; i++) {
        diff = abs(y_ref[i] - y[i]);
        if (diff > tolerance)
            error += diff;
    }
    printf("total error: %f\n", error);
    printf("  reference: %f at (42)\n", y_ref[42]);
    printf("     result: %f at (42)\n", y[42]);

    return 0;
}
