#include <iostream>
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "FFT.h"
using namespace cv;

int main() {
	double cpu_time, gpu_time;
	int cpu_mem_usage, gpu_mem_usage;
	int rows = 4096;
	int cols = 4096;
	fft(rows, cols, cpu_time, cpu_mem_usage, gpu_time, gpu_mem_usage);

	std::cout << "CPU time: " << cpu_time << " seconds" << std::endl;
	std::cout << "CPU memory usage: " << cpu_mem_usage << " MB" << std::endl;
	std::cout << "GPU time: " << gpu_time << " seconds" << std::endl;
	std::cout << "GPU memory usage: " << gpu_mem_usage << " MB" << std::endl;

}