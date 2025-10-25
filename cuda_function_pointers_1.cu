template <typename P>
struct cudaCallableFunctionPointer
{
public:
  cudaCallableFunctionPointer(P* f_)
  {
    P* host_ptr = (P*)malloc(sizeof(P));
    cudaMalloc((void**)&ptr, sizeof(P));

    cudaMemcpyFromSymbol(host_ptr, *f_, sizeof(P));
    cudaMemcpy(ptr, host_ptr, sizeof(P), cudaMemcpyHostToDevice);
    
    cudaFree(host_ptr)
  }

  ~cudaCallableFunctionPointer()
  {
    cudaFree(ptr);
  }

  P* ptr;
};


__device__ double func1(double x)
{
    return x + 1.0f;
}

typedef double (*func)(double x);
__device__ func f_ = func1;

__global__ void test_kernel(func* f)
{
    double x = (*f)(2.0);
    printf("%g\n", x);
}



int main()
{
    cudaCallableFunctionPointer<func> f(&f_);
    test_kernel << < 1, 1 >> > (f.ptr);
}
