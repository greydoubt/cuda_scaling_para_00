//your function pointer type - returns unsigned char, takes parameters of type unsigned char and float
typedef unsigned char(*pointFunction_t)(unsigned char, float);

//some device function to be pointed to
__device__ unsigned char
Threshold(unsigned char in, float thresh)
{
   ...
}

//pComputeThreshold is a device-side function pointer to your __device__ function
__device__ pointFunction_t pComputeThreshold = Threshold;
//the host-side function pointer to your __device__ function
pointFunction_t h_pointFunction;

//in host code: copy the function pointers to their host equivalent
cudaMemcpyFromSymbol(&h_pointFunction, pComputeThreshold, sizeof(pointFunction_t))

//You can then pass the h_pointFunction as a parameter to your kernel, which can use it to call your __device__ function.

//your kernel taking your __device__ function pointer as a parameter
__global__ void kernel(pointFunction_t pPointOperation)
{
    unsigned char tmp;
    ...
    tmp = (*pPointOperation)(tmp, 150.0)
    ...
}

//invoke the kernel in host code, passing in your host-side __device__ function pointer
kernel<<<...>>>(h_pointFunction);
