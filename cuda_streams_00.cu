#include <stdio.h>

int main(){
  cudaStream_t stream;       // CUDA streams are of type `cudaStream_t`.
  cudaStreamCreate(&stream); // Note that a pointer must be passed to `cudaCreateStream`.
  someKernel<<<number_of_blocks, threads_per_block, 0, stream>>>(); // `stream` is passed as 4th EC argument.
  cudaStreamDestroy(stream); // Note that a value, not a pointer, is passed to `cudaDestroyStream`.
}
