#include <stdio.h> 
#include <cuda_runtime.h> 
#define RADIUS 2 
__constant__ int dwgt[2*RADIUS+1];

__global__ void stencil1D(int *in, int *out, const int n) {
  int tid = threadIdx.x
  if (tid < n) {
    int result = 0; 
    for (int i = -RADIUS; i <= RADIUS; i++) {
      int idx = tid + i;
      if (idx >= 0 && idx < n) {
        result += in[idx] * dwgt[i + RADIUS]; 
      }
    }
    out[tid] = result; 
  }
}

int main() {
  const int n = 8, width = 2*RADIUS+1; 
  int host[n] = {1, 2, 3, 4, 5, 6, 7, 8};
  int *input, *output; 
  int hwgt[width] = {1, 1, 1, 1, 1};
  cudaMalloc((void**)&input, n * sizeof(int));
  cudaMalloc((void**)&output, n * sizeof(int));
  cudaMemcpy(input, host, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(dwgt, hwgt, width * sizeof(int));
  stencil1D<<<1, n>>>(input, output, n);
  cudaMemcpy(host, output, n * sizeof(int), cudaMemcpyDeviceToHost);
  printf("Input List: ");
  for (int i = 0; i < n; i++)
    printf("%d ", input[i]);
    printf("\n");
  printf("Output List: ");
  for (int i = 0; i < n; i++)
    printf("%d ", input[i]);
    printf("\n");

  cudaFree(input); 
  cudaFree(output);
  return 0; 
}
