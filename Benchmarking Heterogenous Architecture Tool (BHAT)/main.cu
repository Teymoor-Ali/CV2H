#include <iostream>
#include <opencv2/imgproc.hpp>
#include "FFT.h"
#include "STREAM.cuh"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

using namespace cv;

int main(int argc, char* argv[]) {
	if (argc < 2) {
		std::cout << "Usage: " << argv[0] << " <function> [args...]" << std::endl;
		return 1;
	}

	std::string function = argv[1];

	if (function == "FFT") {
		if (argc != 3) {
			std::cout << "Usage: " << argv[0] << " FFT <size>" << std::endl;
			return 1;
		}

		int size = std::stoi(argv[2]);
		double cpu_time, gpu_time;
		int cpu_mem_usage, gpu_mem_usage;
		fft(size, size, cpu_time, cpu_mem_usage, gpu_time, gpu_mem_usage);

		std::cout << "CPU time: " << cpu_time << " seconds" << std::endl;
		std::cout << "CPU memory usage: " << cpu_mem_usage << " MB" << std::endl;
		std::cout << "GPU time: " << gpu_time << " seconds" << std::endl;
		std::cout << "GPU memory usage: " << gpu_mem_usage << " MB" << std::endl;
	}
	else if (function == "STREAM") {
		if (argc != 3) {
			std::cout << "Usage: " << argv[0] << " STREAM <num_streams>" << std::endl;
			return 1;
		}

		int num_streams = std::stoi(argv[2]);
		STREAM(num_streams);
	}
	else {
		std::cout << "Unknown function: " << function << std::endl;
		return 1;
	}

	return 0;
}
