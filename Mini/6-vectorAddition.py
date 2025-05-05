cuda_code = """
#include <iostream>
using namespace std;

_global_ void vectorAdd(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int n;
    cout << "Enter the size of vectors: ";
    cin >> n;

    int *a = new int[n];
    int *b = new int[n];
    int *c = new int[n];

    cout << "Enter elements of vector a:\\n";
    for (int i = 0; i < n; ++i) cin >> a[i];

    cout << "Enter elements of vector b:\\n";
    for (int i = 0; i < n; ++i) cin >> b[i];

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(int));
    cudaMalloc(&d_b, n * sizeof(int));
    cudaMalloc(&d_c, n * sizeof(int));

    cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "CUDA error: " << cudaGetErrorString(err) << endl;
        return 1;
    }

    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "Result vector:\\n";
    for (int i = 0; i < n; i++) cout << c[i] << " ";
    cout << endl;

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] a; delete[] b; delete[] c;

    return 0;
}
"""

with open("vector_add.cu", "w") as f:
    f.write(cuda_code)

# Compile and run
!nvcc -arch=sm_70 vector_add.cu -o vector_add
!./vector_add