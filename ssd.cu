// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

#include <sys/time.h>

long getMicrotime() {
	struct timeval currentTime;
	gettimeofday(&currentTime, NULL);
	return currentTime.tv_sec * (int)1e6 + currentTime.tv_usec;
}

extern "C" __global__ void ssd_cuda(const int *window, const int *target, int *out,
					const int window_width, const int window_height,
					const int target_width, const int target_height,
					const int out_width, const int out_height) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
   	int x = i % out_width;
    int y = i / out_width;
	int sum = 0;
	for(int i = 0; i < window_height; ++i) {
		for(int j = 0; j < window_width; ++j) {
			int diff = window[i * window_width + j] - target[(i + y)* target_width + j + x];
			sum += diff * diff;
		}
	}
	out[y * out_width + x] = sum;
}

extern "C" void ssd(const void *window_, const void *target_, void *out_,
					const int window_width, const int window_height,
					const int target_width, const int target_height,
					const int out_width, const int out_height) {
	//long second = getMicrotime();
	/*cudaSetDevice(0);
	cudaDeviceSynchronize();
	cudaThreadSynchronize();*/
	/*printf("in function\n");
	printf("int size: %d\n", sizeof(int));*/

	const int *window = (int *) window_;
	const int *target = (int *) target_;
	int *out = (int *) out_;

	/*printf("\nsize: %d, %d, %d, %d, %d, %d\n", window_width, window_height,
			target_width, target_height, out_width, out_height);*/

	cudaError_t err = cudaSuccess;
	size_t size_window = window_width * window_height * sizeof(int);
	size_t size_target = target_width * target_height * sizeof(int);
	size_t size_out = out_width * out_height * sizeof(int);

	//printf("size: %d, %d, %d\n", size_window, size_target, size_out);

	//printf("\n\n%ld\n\n", getMicrotime() - second);

	int *d_window = NULL;
    err = cudaMalloc((void **)&d_window, size_window);
    if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device d_window (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
    }
	int *d_target = NULL;
    err = cudaMalloc((void **)&d_target, size_target);
    if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device d_target (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
    }
	int *d_out = NULL;
	err = cudaMalloc((void **)&d_out, size_out);
    if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device d_out (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(d_window, window, size_window, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
		exit(EXIT_FAILURE);
    }
	err = cudaMemcpy(d_target, target, size_target, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
		exit(EXIT_FAILURE);
    }

	int threadsPerBlock = out_width;
    int blocksPerGrid = out_height;
    // printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	ssd_cuda<<<blocksPerGrid, threadsPerBlock>>>(d_window, d_target, d_out,
												window_width, window_height,
												target_width, target_height,
												out_width, out_height);

	err = cudaMemcpy(out, d_out, size_out, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
		exit(EXIT_FAILURE);
	}

	/*for(int i = 0; i < out_width * out_height; ++i) {
		printf("%d\t", out[i]);
	}*/

	err = cudaFree(d_window);
    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }
	err = cudaFree(d_target);
    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }
	err = cudaFree(d_out);
    if (err != cudaSuccess) {
        exit(EXIT_FAILURE);
    }

	//printf("\nDone\n");
}

int main() {
	int window_width = 5;
	int window_height = 5;
	int target_width = 20;
	int target_height = 10;
	int out_width = 16;
	int out_height = 6;

	int window[5 * 5] = {0};
	int target[20 * 10] = {0};
	int out[16 * 6] = {0};

	for(int i = 0; i < target_width * target_height; ++i) {
		target[i] = i;
	}

	ssd(window, target, out, window_width, window_height, target_width, target_height,
		out_width, out_height);

	for(int i = 0; i < out_width * out_height; ++i) {
		printf("%d\t", out[i]);
	}

	return 0;
}
