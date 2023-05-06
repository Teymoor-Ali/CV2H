#ifndef STREAM_HEADER_H
#define STREAM_HEADER_H


#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <math.h>
#include <algorithm>
#include <numeric>

//#define N (100)
#define THREADS_PER_BLOCK 256
#define ITERATIONS 4

__global__ void copy_kernel(double* dst, double* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

__global__ void add_kernel(double* dst, double* a, double* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = a[idx] + b[idx];
    }
}

__global__ void scale_kernel(double* dst, double* src, double scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = scalar * src[idx];
    }
}

__global__ void triad_kernel(double* dst, double* a, double* b, double* c, double scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = a[idx] + scalar * b[idx] + c[idx];
    }
}
void STREAM(int N) {
    double* a, * b, * c, * d;
    double scalar = 2.0f;
    size_t bytes = N * sizeof(double);

    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);
    cudaMallocManaged(&d, bytes);

    for (int i = 0; i < N; i++) {
        a[i] = (double)i;
        b[i] = (double)(N - i);
        c[i] = (double)(i % 3);
    }

    dim3 grid((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float copy_time = 0.0f, add_time = 0.0f, scale_time = 0.0f, triad_time = 0.0f;

    for (int i = 0; i < ITERATIONS; i++) {
        cudaEventRecord(start);
        copy_kernel << <grid, THREADS_PER_BLOCK >> > (d, a, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        copy_time += elapsed_time;
    }
    printf("Copy time - Avg: %f ms, Min: %f ms, Max: %f ms, Throughput: %f GB/s\n",
        copy_time / ITERATIONS,
        copy_time / ITERATIONS,
        copy_time / ITERATIONS,
        (bytes * 2.0f * (float)N / (copy_time / (ITERATIONS * 1000.0f)) / (1024.0f * 1024.0f * 1024)));

    for (int i = 0; i < ITERATIONS; i++) {
        cudaEventRecord(start);
        add_kernel << <grid, THREADS_PER_BLOCK >> > (d, a, b, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        add_time += elapsed_time;
    }
    printf("Add time - Avg: %f ms, Min: %f ms, Max: %f ms, Throughput: %f GB/s\n",
        add_time / ITERATIONS,
        add_time / ITERATIONS,
        add_time / ITERATIONS,
        (bytes * 3.0f * (float)N / (add_time / (ITERATIONS * 1000.0f)) / (1024.0f * 1024.0f * 1024)));
    for (int i = 0; i < ITERATIONS; i++) {
        cudaEventRecord(start);
        scale_kernel << <grid, THREADS_PER_BLOCK >> > (d, a, scalar, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        scale_time += elapsed_time;
    }
    printf("Scale time - Avg: %f ms, Min: %f ms, Max: %f ms, Throughput: %f GB/s\n",
        scale_time / ITERATIONS,
        scale_time / ITERATIONS,
        scale_time / ITERATIONS,
        (bytes * 2.0f * (float)N / (scale_time / (ITERATIONS * 1000.0f)) / (1024.0f * 1024.0f * 1024)));

    for (int i = 0; i < ITERATIONS; i++) {
        cudaEventRecord(start);
        triad_kernel << <grid, THREADS_PER_BLOCK >> > (d, a, b, c, scalar, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        triad_time += elapsed_time;
    }
    printf("Triad time - Avg: %f ms, Min: %f ms, Max: %f ms, Throughput: %f GB/s\n",
        triad_time / ITERATIONS,
        triad_time / ITERATIONS,
        triad_time / ITERATIONS,
        (bytes * 4.0f * (float)N / (triad_time / (ITERATIONS * 1000.0f)) / (1024.0f * 1024.0f * 1024)));

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(d);
}
#endif // STREAM_HEADER_H