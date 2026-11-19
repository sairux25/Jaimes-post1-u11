#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define TILE 16

// Revisa el error del ultimo lanzamiento o llamada CUDA y detiene el programa si falla
#define CHECK_CUDA(msg)                                                  \
    do {                                                                 \
        cudaError_t err = cudaGetLastError();                            \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "Error CUDA en %s: %s\n", msg,               \
                    cudaGetErrorString(err));                            \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

// Kernel naive: cada thread calcula C[row][col] leyendo A y B desde memoria global
__global__ void matMulNaive(const float *A, const float *B,
                            float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row*N + k] * B[k*N + col];
        C[row*N + col] = sum;
    }
}

// Kernel con tiling: cada bloque carga tiles de A y B en shared memory
__global__ void matMulTiled(const float *A, const float *B,
                            float *C, int N) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
        // Cargar tile de A en shared memory
        if (row < N && t*TILE + threadIdx.x < N)
            sA[threadIdx.y][threadIdx.x] =
                A[row*N + t*TILE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        // Cargar tile de B en shared memory
        if (col < N && t*TILE + threadIdx.y < N)
            sB[threadIdx.y][threadIdx.x] =
                B[(t*TILE + threadIdx.y)*N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads(); // Esperar a que el tile este cargado

        for (int k = 0; k < TILE; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads(); // Esperar antes del siguiente tile
    }

    if (row < N && col < N) C[row*N + col] = sum;
}

// Multiplicacion de matrices en CPU (referencia)
void matMulCPU(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
                sum += A[i*N + k] * B[k*N + j];
            C[i*N + j] = sum;
        }
}

// Devuelve la mayor diferencia absoluta entre dos matrices
float maxError(const float *X, const float *Y, int n) {
    float maxErr = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(X[i] - Y[i]);
        if (diff > maxErr) maxErr = diff;
    }
    return maxErr;
}

// Ejecuta el benchmark completo para una matriz de N x N
void benchmark(int N) {
    size_t bytes = (size_t)N * N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_cpu = (float*)malloc(bytes);
    float *h_C_naive = (float*)malloc(bytes);
    float *h_C_tiled = (float*)malloc(bytes);

    // Inicializar matrices con valores entre 0 y 1
    srand(42);
    for (int i = 0; i < N*N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    matMulCPU(h_A, h_B, h_C_cpu, N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    CHECK_CUDA("cudaMalloc");

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    CHECK_CUDA("cudaMemcpy H2D");

    // Lanzamiento: grid 2D de tiles
    dim3 block(TILE, TILE);
    dim3 grid((N+TILE-1)/TILE, (N+TILE-1)/TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Lanzamiento previo de cada kernel para que la carga inicial no entre en la medicion
    matMulNaive<<<grid, block>>>(d_A, d_B, d_C, N);
    matMulTiled<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    CHECK_CUDA("lanzamiento previo");

    // --- Kernel naive ---
    cudaEventRecord(start);
    matMulNaive<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    CHECK_CUDA("matMulNaive");
    float naiveMs = 0;
    cudaEventElapsedTime(&naiveMs, start, stop);
    cudaMemcpy(h_C_naive, d_C, bytes, cudaMemcpyDeviceToHost);

    // --- Kernel con tiling ---
    cudaEventRecord(start);
    matMulTiled<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    CHECK_CUDA("matMulTiled");
    float tiledMs = 0;
    cudaEventElapsedTime(&tiledMs, start, stop);
    cudaMemcpy(h_C_tiled, d_C, bytes, cudaMemcpyDeviceToHost);
    CHECK_CUDA("cudaMemcpy D2H");

    printf("N = %d\n", N);
    printf("  Naive: %.3f ms  (error max vs CPU: %.2e)\n",
           naiveMs, maxError(h_C_naive, h_C_cpu, N*N));
    printf("  Tiled: %.3f ms  (error max vs CPU: %.2e)\n",
           tiledMs, maxError(h_C_tiled, h_C_cpu, N*N));
    printf("  Speedup tiling: %.2fx\n", naiveMs / tiledMs);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_naive); free(h_C_tiled);
}

int main() {
    benchmark(512);
    benchmark(1024);
    return 0;
}
