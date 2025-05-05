cuda_code = """
#include <iostream>
using namespace std;

_global_ void matrixMul(int *A, int *B, int *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int value = 0;
        for (int k = 0; k < n; k++) {
            value += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = value;
    }
}

int main() {
    int n;
    cout << "Enter the size of the matrices (n x n): ";
    cin >> n;

    int *A = new int[n * n];
    int *B = new int[n * n];
    int *C = new int[n * n];

    cout << "Enter elements of matrix A:\\n";
    for (int i = 0; i < n * n; i++) cin >> A[i];

    cout << "Enter elements of matrix B:\\n";
    for (int i = 0; i < n * n; i++) cin >> B[i];

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n * n * sizeof(int));
    cudaMalloc(&d_B, n * n * sizeof(int));
    cudaMalloc(&d_C, n * n * sizeof(int));

    cudaMemcpy(d_A, A, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((n + 15) / 16, (n + 15) / 16);
    matrixMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);

    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "Result matrix C:\\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << C[i * n + j] << " ";
        }
        cout << endl;
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] A; delete[] B; delete[] C;
    return 0;
}
"""


# Save CUDA code to a file
with open("matrix_mul.cu", "w") as f:
    f.write(cuda_code)

# Compile the CUDA code for the appropriate architecture (sm_70 for Tesla T4)
!nvcc -arch=sm_70 matrix_mul.cu -o matrix_mul

# Run the compiled program
!./matrix_mul