#pragma once
#ifndef STREAM_HEADER_H
#define STREAM_HEADER_H



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <numeric>
#include <chrono>

void copy_kernel(double* dst, double* src, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

void add_kernel(double* dst, double* a, double* b, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = a[i] + b[i];
    }
}

void scale_kernel(double* dst, double* src, double scalar, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = scalar * src[i];
    }
}

void triad_kernel(double* dst, double* a, double* b, double* c, double scalar, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = a[i] + scalar * b[i] + c[i];
    }
}

void STREAM(int N) {
    double* a, * b, * c, * d;
    double scalar = 2.0f;
    size_t bytes = N * sizeof(double);

    a = (double*)malloc(bytes);
    b = (double*)malloc(bytes);
    c = (double*)malloc(bytes);
    d = (double*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        a[i] = (double)i;
        b[i] = (double)(N - i);
        c[i] = (double)(i % 3);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 4; i++) {
        copy_kernel(d, a, N);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto copy_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
    printf("Copy time - Avg: %f ms, Min: %f ms, Max: %f ms, Throughput: %f GB/s\n",
        copy_time / 4.0,
        copy_time / 4.0,
        copy_time / 4.0,
        (bytes * 2.0 * (double)N / (copy_time / 1000.0) / (1024.0 * 1024.0 * 1024)));

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 4; i++) {
        add_kernel(d, a, b, N);
    }
    stop = std::chrono::high_resolution_clock::now();
    auto add_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
    printf("Add time - Avg: %f ms, Min: %f ms, Max: %f ms, Throughput: %f GB/s\n",
        add_time / 4.0,
        add_time / 4.0,
        add_time / 4.0,
        (bytes * 3.0 * (double)N / (add_time / 1000.0) / (1024.0 * 1024.0 * 1024)));

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 4; i++) {
        scale_kernel(d, a, scalar, N);
    }
    stop = std::chrono::high_resolution_clock::now();
    auto scale_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
    printf("Scale time - Avg: %f ms, Min: %f ms, Max: %f ms, Throughput: %f GB/s\n",
        scale_time / 4.0,
        scale_time / 4.0,
        scale_time / 4.0,
        (bytes * 2.0 * (double)N / (scale_time / 1000.0) / (1024.0 * 1024.0 * 1024)));
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 4; i++) {
        triad_kernel(d, a, b, c, scalar, N);
    }
    stop = std::chrono::high_resolution_clock::now();
    auto triad_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
    printf("Triad time - Avg: %f ms, Min: %f ms, Max: %f ms, Throughput: %f GB/s\n",
        triad_time / 4.0,
        triad_time / 4.0,
        triad_time / 4.0,
        (bytes * 4.0 * (double)N / (triad_time / 1000.0) / (1024.0 * 1024.0 * 1024)));

    free(a);
    free(b);
    free(c);
    free(d);
}
#endif // FFT_HEADER_H