#include <iostream>
#include <cuda_runtime.h>

__global__ void addOneKernel(int *device_a, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        device_a[idx] += 1;
    }
}

void initializeOnHost(int *host_a, int N) {
    for (int i = 0; i < N; ++i) {
        host_a[i] = i;  // Initialize with some values, e.g., 0, 1, 2, ...
    }
}

void verifyOnHost(int *host_a, int N) {
    for (int i = 0; i < N; ++i) {
        if (host_a[i] != i + 1) {  // Check if each element was incremented by 1
            std::cerr << "Error: host_a[" << i << "] = " << host_a[i] << " (expected " << i + 1 << ")\n";
            return;
        }
    }
    std::cout << "Verification successful! All elements are incremented by 1.\n";
}

int main() {
    const int N = 1024;           // Number of elements
    const size_t size = N * sizeof(int);  // Size of the array in bytes
    int *host_a, *device_a;

    // Allocate memory
    cudaMalloc(&device_a, size);         // Device memory
    cudaMallocHost(&host_a, size);       // Pinned host memory

    // Initialize host array
    initializeOnHost(host_a, N);

    // Copy data from host to device
    cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);

    // Define kernel launch parameters
    int blocks = (N + 255) / 256;   // Calculate number of blocks
    int threads = 256;               // Number of threads per block

    // Launch the kernel to add 1 to each element in device_a
    addOneKernel<<<blocks, threads>>>(device_a, N);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy the results back to host
    cudaMemcpy(host_a, device_a, size, cudaMemcpyDeviceToHost);

    // Verify the result
    verifyOnHost(host_a, N);

    // Free allocated memory
    cudaFree(device_a);
    cudaFreeHost(host_a);

    return 0;
}
