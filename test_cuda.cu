// Directly run cuda code in jupyter notebook

// !pip install nvcc4jupyter
//%load_ext nvcc4jupyter
//%%cuda


#include <stdio.h>

__global__ void helloFromGPU() {
    printf("Hello from GPU thread %d, block %d\n", threadIdx.x, blockIdx.x);
}

int main() {
    printf("Hello from CPU\n");
    helloFromGPU<<<2, 5>>>();
    cudaDeviceSynchronize();
    return 0;
}
