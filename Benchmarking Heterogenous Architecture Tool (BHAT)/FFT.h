#pragma once
#ifndef FFT_HEADER_H
#define FFT_HEADER_H

#include <iostream>
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>


using namespace cv;

void fft(const Mat& img, double& cpu_time, int& cpu_mem_usage, double& gpu_time, int& gpu_mem_usage) {
    // Set up CPU FFT
    Mat padded;
    int m = getOptimalDFTSize(img.rows);
    int n = getOptimalDFTSize(img.cols);
    copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat complexImg;
    merge(planes, 2, complexImg);
    TickMeter tm_cpu;
    tm_cpu.start();
    dft(complexImg, complexImg);
    tm_cpu.stop();
    cpu_time = tm_cpu.getTimeSec();
    cpu_mem_usage = complexImg.elemSize() * complexImg.total() / 1024 / 1024;

    // Set up GPU FFT
    cuda::GpuMat gpuImg(img);
    cuda::GpuMat gpuComplexImg;
    cuda::GpuMat gpuPlanes[2];
    cuda::Stream stream;
    gpuPlanes[0] = cuda::GpuMat(gpuImg.size(), CV_32FC1);
    gpuPlanes[1] = cuda::GpuMat(gpuImg.size(), CV_32FC1);
    gpuComplexImg = cuda::GpuMat(gpuImg.size(), CV_32FC2);
    gpuImg.convertTo(gpuPlanes[0], CV_32FC1);
    gpuPlanes[1].setTo(0);
    cuda::merge(gpuPlanes, 2, gpuComplexImg);
    TickMeter tm_gpu;
    tm_gpu.start();
    cuda::dft(gpuComplexImg, gpuComplexImg, gpuComplexImg.size(), 0, stream);
    tm_gpu.stop();
    gpu_time = tm_gpu.getTimeSec();
    gpu_mem_usage = gpuComplexImg.elemSize() * gpuComplexImg.size().area() / 1024 / 1024;
}

void fft(int rows, int cols, double& cpu_time, int& cpu_mem_usage, double& gpu_time, int& gpu_mem_usage) {
    // Create a 4k image
    Mat img(rows, cols, CV_8UC1);
    img.setTo(Scalar(128)); // Set all pixels to gray

    fft(img, cpu_time, cpu_mem_usage, gpu_time, gpu_mem_usage);
}

#endif // FFT_HEADER_H
