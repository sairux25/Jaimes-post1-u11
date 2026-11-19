#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define N (1 << 24)        // 16M elementos, igual que vectorAdd.cu
#define NSTREAMS 4
#define CHUNK (N / NSTREAMS)

__global__ void vectorAddChunk(const float *A, const float *B,
                               float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] + B[idx];
}

// Suma de vectores en CPU (referencia)
void vectorAddCPU(const float *A, const float *B,
                  float *C, int n) {
    for (int i = 0; i < n; i++) C[i] = A[i] + B[i];
}

// Cuenta los elementos que difieren del resultado de CPU
int countErrors(const float *gpu, const float *cpu, int n) {
    int errors = 0;
    for (int i = 0; i < n; i++)
        if (fabs(gpu[i] - cpu[i]) > 1e-4f) errors++;
    return errors;
}

int main() {
    size_t bytes      = N * sizeof(float);
    size_t chunkBytes = CHUNK * sizeof(float);

    // Memoria host "pinned" (page-locked): obligatoria para que
    // cudaMemcpyAsync se ejecute realmente de forma asincrona
    float *h_A, *h_B, *h_C;
    cudaHostAlloc(&h_A, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_B, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_C, bytes, cudaHostAllocDefault);

    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i * 0.5f;
        h_B[i] = (float)i * 1.5f;
    }

    // Resultado de referencia en CPU, igual que en vectorAdd.cu
    float *h_C_cpu = (float*)malloc(bytes);
    vectorAddCPU(h_A, h_B, h_C_cpu, N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int blockSize = 256;
    int chunkGrid = (CHUNK + blockSize - 1) / blockSize;

    // --- Baseline sincrono: una sola transferencia H2D, un kernel, una D2H ---
    cudaEventRecord(start);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    vectorAddChunk<<<(N + blockSize - 1) / blockSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float syncMs = 0;
    cudaEventElapsedTime(&syncMs, start, stop);
    printf("Sincrono (H2D + kernel + D2H completos): %.2f ms\n", syncMs);
    printf("Errores sincrono: %d\n", countErrors(h_C, h_C_cpu, N));

    // Limpiar el resultado para que la version con streams no reutilice el anterior
    memset(h_C, 0, bytes);
    cudaMemset(d_C, 0, bytes);
    cudaDeviceSynchronize();

    // --- Version con streams: solapa transferencia y computo por chunks ---
    cudaStream_t streams[NSTREAMS];
    for (int s = 0; s < NSTREAMS; s++) cudaStreamCreate(&streams[s]);

    cudaEventRecord(start);
    for (int s = 0; s < NSTREAMS; s++) {
        size_t off = (size_t)s * CHUNK;
        cudaMemcpyAsync(d_A + off, h_A + off, chunkBytes,
                        cudaMemcpyHostToDevice, streams[s]);
        cudaMemcpyAsync(d_B + off, h_B + off, chunkBytes,
                        cudaMemcpyHostToDevice, streams[s]);
        vectorAddChunk<<<chunkGrid, blockSize, 0, streams[s]>>>(
            d_A + off, d_B + off, d_C + off, CHUNK);
        cudaMemcpyAsync(h_C + off, d_C + off, chunkBytes,
                        cudaMemcpyDeviceToHost, streams[s]);
    }
    for (int s = 0; s < NSTREAMS; s++) cudaStreamSynchronize(streams[s]);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float asyncMs = 0;
    cudaEventElapsedTime(&asyncMs, start, stop);
    printf("Con streams (%d, solapado): %.2f ms\n", NSTREAMS, asyncMs);
    printf("Errores con streams: %d\n", countErrors(h_C, h_C_cpu, N));
    printf("Reduccion: %.1f%%\n", 100.0 * (syncMs - asyncMs) / syncMs);

    for (int s = 0; s < NSTREAMS; s++) cudaStreamDestroy(streams[s]);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C);
    free(h_C_cpu);
    return 0;
}
