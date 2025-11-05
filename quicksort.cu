#include <stdio.h>
#include <math.h>
#include <cuda.h>

#define N 16          // Number of elements
#define TPB 8         // Threads per block
#define MIN_PART 16   // Use insertion sort for small partitions

// -----------------------------
// Device functions
// -----------------------------

// Insertion sort for small segments
__device__ void insertionSort(int *arr, int left, int right) {
    for (int i = left + 1; i <= right; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= left && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// Partition function for quicksort
__device__ int partition(int *arr, int left, int right) {
    int pivot = arr[right];
    int i = left - 1;

    for (int j = left; j < right; j++) {
        if (arr[j] < pivot) {
            i++;
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }

    int temp = arr[i + 1];
    arr[i + 1] = arr[right];
    arr[right] = temp;

    return i + 1;
}

// Kernel: process each partition
__global__ void quicksortKernel(int *arr, int *L, int *R, int *newL, int *newR, int nTasks, int *nextCount) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nTasks) return;

    int left = L[id];
    int right = R[id];

    if (left < right) {
        // Use insertion sort for small partitions
        if (right - left + 1 <= MIN_PART) {
            insertionSort(arr, left, right);
            return;
        }

        int p = partition(arr, left, right);

        // Add new tasks atomically
        if (p - 1 > left) {
            int idx = atomicAdd(nextCount, 1);
            newL[idx] = left;
            newR[idx] = p - 1;
        }
        if (p + 1 < right) {
            int idx = atomicAdd(nextCount, 1);
            newL[idx] = p + 1;
            newR[idx] = right;
        }
    }
}

// -----------------------------
// Host code
// -----------------------------

int main() {
    int h_arr[N] = {24, 17, 85, 13, 9, 54, 76, 45, 4, 63, 21, 33, 89, 12, 99, 1};

    // Allocate memory on device
    int *d_arr;
    cudaMalloc(&d_arr, N * sizeof(int));
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    int *L, *R, *newL, *newR, *nextCount;
    cudaMalloc(&L, N * sizeof(int));
    cudaMalloc(&R, N * sizeof(int));
    cudaMalloc(&newL, N * sizeof(int));
    cudaMalloc(&newR, N * sizeof(int));
    cudaMalloc(&nextCount, sizeof(int));

    // Initialize first task
    int h_L[1] = {0}, h_R[1] = {N - 1};
    cudaMemcpy(L, h_L, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(R, h_R, sizeof(int), cudaMemcpyHostToDevice);

    // Timing setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Quicksort iterations
    int nTasks = 1;
    while (nTasks > 0) {
        cudaMemset(nextCount, 0, sizeof(int));

        int blocks = (nTasks + TPB - 1) / TPB;
        quicksortKernel<<<blocks, TPB>>>(d_arr, L, R, newL, newR, nTasks, nextCount);
        cudaDeviceSynchronize();

        cudaMemcpy(&nTasks, nextCount, sizeof(int), cudaMemcpyDeviceToHost);

        // Swap task buffers
        int *tmpL = L; L = newL; newL = tmpL;
        int *tmpR = R; R = newR; newR = tmpR;
    }

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy result back
    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Complexity calculations
    double log2n = log2((double)N);
    double timeComplexity = ((double)N * log2n) / TPB;
    double costComplexity = (double)N * log2n;

    // Print results
    printf("Sorted Array: ");
    for (int i = 0; i < N; i++)
        printf("%d ", h_arr[i]);
    printf("\n\n===== COMPLEXITY ANALYSIS =====\n");
    printf("Time Complexity : O((N log N)/P) = %.2f units\n", timeComplexity);
    printf("Cost Complexity : O(N log N) = %.2f units\n", costComplexity);
    printf("Execution Time  : %.5f ms\n", ms);

    // Cleanup
    cudaFree(d_arr);
    cudaFree(L); cudaFree(R);
    cudaFree(newL); cudaFree(newR);
    cudaFree(nextCount);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
