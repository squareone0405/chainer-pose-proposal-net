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
					const int *window_width, const int *window_height,
					const int *target_width, const int *target_height,
					const int *window_offset, const int *target_offset, 
					const int out_width, const int out_height, const int num) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int z = tid / (out_width * out_height);
    int y = tid / out_width - z * out_height;
	int x = tid % out_width;
	int sum = 0;
	const int* window_ptr = NULL;
	window_ptr = window + window_offset[z];
	const int* target_ptr = NULL;
	target_ptr = target + target_offset[z] + y * target_width[z] + x;
	for(int i = 0; i < window_height[z]; ++i) {
		for(int j = 0; j < window_width[z]; ++j) {
			int diff = *window_ptr - *target_ptr;
			sum += diff * diff;
			window_ptr++;
			target_ptr++;
		}
		target_ptr += target_width[z] - window_width[z];
	}
	out[tid] = sum;
}

extern "C" void ssd(const void *window_, const void *target_, void *out_,
					const void *window_width_, const void *window_height_,
					const void *target_width_, const void *target_height_,
					const int out_width, const int out_height, const int num) {
	long second = getMicrotime();
	/*cudaSetDevice(0);
	cudaDeviceSynchronize();
	cudaThreadSynchronize();*/

	const int *window = (int *) window_;
	const int *target = (int *) target_;
	const int *window_width = (int *) window_width_;
	const int *window_height = (int *) window_height_;
	const int *target_width = (int *) target_width_;
	const int *target_height = (int *) target_height_;
	int *out = (int *) out_;

	/*printf("\nsize: %d, %d, %d, %d, %d, %d\n", window_width[0], window_height[0],
			target_width[0], target_height[0], out_width, out_height);*/

	cudaError_t err = cudaSuccess;

	size_t size_window = 0;
	size_t size_target = 0;
	int *window_offset = (int *)malloc(sizeof(int) * num);
	int *target_offset = (int *)malloc(sizeof(int) * num);
	for(int i = 0; i < num; ++i) {
		window_offset[i] = size_window;
		target_offset[i] = size_target;
		size_window += window_width[i] * window_height[i];
		size_target += target_width[i] * target_height[i];
	}
	size_window *= sizeof(int);
	size_target *= sizeof(int);
	size_t size_out = out_width * out_height * num * sizeof(int);
	size_t size_num = num * sizeof(int);

	//printf("size: %d, %d, %d\n", size_window, size_target, size_out);

	int *d_window = NULL;
    err = cudaMalloc((void **)&d_window, size_window);
    if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device d_window (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
    }

	//printf("malloc 1\n%ld\n", getMicrotime() - second);

	int *d_target = NULL;
    err = cudaMalloc((void **)&d_target, size_target);
    if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device d_target (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
    }

	//printf("malloc 2\n%ld\n", getMicrotime() - second);

	int *d_out = NULL;
	err = cudaMalloc((void **)&d_out, size_out);
    if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device d_out (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//printf("malloc 3\n%ld\n", getMicrotime() - second);

	int *d_window_width = NULL;
	err = cudaMalloc((void **)&d_window_width, size_num);
    if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device d_window_width (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	int *d_window_height = NULL;
	err = cudaMalloc((void **)&d_window_height, size_num);
    if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device d_window_height (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	int *d_target_width = NULL;
	err = cudaMalloc((void **)&d_target_width, size_num);
    if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device d_target_width (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	int *d_target_height = NULL;
	err = cudaMalloc((void **)&d_target_height, size_num);
    if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device d_target_height (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	int *d_window_offset = NULL;
	err = cudaMalloc((void **)&d_window_offset, size_num);
    if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device d_window_offset (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	int *d_target_offset = NULL;
	err = cudaMalloc((void **)&d_target_offset, size_num);
    if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device d_target_offset (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//printf("malloc 4\n%ld\n", getMicrotime() - second);

	err = cudaMemcpy(d_window, window, size_window, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
		exit(EXIT_FAILURE);
    }
	err = cudaMemcpy(d_target, target, size_target, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
		exit(EXIT_FAILURE);
    }
	err = cudaMemcpy(d_window_width, window_width, size_num, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
		exit(EXIT_FAILURE);
    }
	err = cudaMemcpy(d_window_height, window_height, size_num, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
		exit(EXIT_FAILURE);
    }
	err = cudaMemcpy(d_target_width, target_width, size_num, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
		exit(EXIT_FAILURE);
    }
	err = cudaMemcpy(d_target_height, target_height, size_num, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
		exit(EXIT_FAILURE);
    }
	err = cudaMemcpy(d_window_offset, window_offset, size_num, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
		exit(EXIT_FAILURE);
    }
	err = cudaMemcpy(d_target_offset, target_offset, size_num, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
		exit(EXIT_FAILURE);
    }

	//printf("memcpy\n%ld\n", getMicrotime() - second);

	int threadsPerBlock = out_width * out_height;
    int blocksPerGrid = num;
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	ssd_cuda<<<blocksPerGrid, threadsPerBlock>>>(d_window, d_target, d_out,
												d_window_width, d_window_height,
												d_target_width, d_target_height,
												d_window_offset, d_target_offset, 
												out_width, out_height, num);
	/*printf("out---------------------\n");

	for(int i = 0; i < out_width * out_height * num; ++i) {
		printf("%d\t", d_out[i]);
	}*/

	err = cudaMemcpy(out, d_out, size_out, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
		fprintf(stderr, "(error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_window);
    if (err != cudaSuccess) {
		fprintf(stderr, "(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	err = cudaFree(d_target);
    if (err != cudaSuccess) {
		fprintf(stderr, "(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	err = cudaFree(d_out);
    if (err != cudaSuccess) {
		fprintf(stderr, "(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	free(window_offset);
	free(target_offset);

	//printf("\nDone\n");
	printf("done\n%ld-----------\n", getMicrotime() - second);
}

int main() {
	int window_width[] = {5, 6};
	int window_height[] = {5, 6};
	int target_width[] = {20, 21};
	int target_height[] = {10, 11};
	int out_width = 16;
	int out_height = 6;

	int window[5 * 5 + 6 * 6] = {0};
	int target[20 * 10 + 21 * 11] = {1};
	int out[16 * 6 * 2] = {0};

	for(int i = 0; i < target_width[0] * target_height[0]; ++i) {
		target[i] = i;
	}
	for(int i = 0; i < target_width[1] * target_height[1]; ++i) {
		target[i + target_width[0] * target_height[0]] = 1;
	}

	ssd(window, target, out, window_width, window_height, target_width, target_height,
		out_width, out_height, 2);

	for(int i = 0; i < out_width * out_height * 2; ++i) {
		printf("%d\t", out[i]);
	}

	return 0;
}
