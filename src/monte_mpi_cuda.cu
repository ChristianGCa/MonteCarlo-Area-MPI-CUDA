// To execute: mpirun -np 4 ./build/monte_mpi_cuda 256 10000000 maui.png 100.0
#include <mpi.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <cstring>

#include "png_loader.h"

__device__ __host__ inline bool is_blue_pixel(unsigned char r, unsigned char g, unsigned char b) {
    return (r == 255 && g == 0 && b == 0);
}

__global__ void monte_kernel(const unsigned char* d_img, int width, int height, unsigned long long samples_per_rank, unsigned long long seed, unsigned long long* d_count) {
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long threads_launched = (unsigned long long)gridDim.x * blockDim.x;
    if (tid >= threads_launched) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, (unsigned long long)tid, 0, &state);

    unsigned long long local_count = 0;
    unsigned long long total_pixels = (unsigned long long)width * (unsigned long long)height;

    for (unsigned long long idx_sample = tid; idx_sample < samples_per_rank; idx_sample += threads_launched) {
        unsigned long long r1 = (unsigned long long)curand(&state);
        unsigned long long r2 = (unsigned long long)curand(&state);
        unsigned long long r64 = (r1 << 32) ^ r2;
        unsigned long long pix_idx = r64 % total_pixels;
        unsigned long long base = pix_idx * 3ULL;
        unsigned char r = d_img[base + 0];
        unsigned char g = d_img[base + 1];
        unsigned char b = d_img[base + 2];
        if (is_blue_pixel(r, g, b)) local_count++;
    }
    atomicAdd(d_count, local_count);
}

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 4) {
        if (world_rank == 0) {
            fprintf(stderr, "Uso: %s total_samples threads_per_block caminho_imagem.png area_km2\n", argv[0]);
            fprintf(stderr, "Ex: mpirun -np 4 ./monte_mpi_cuda 10000000 256 image.png 288.98\n");
        }
        MPI_Finalize();
        return 1;
    }

    unsigned long long total_samples_requested = strtoull(argv[2], NULL, 10);
    int threads_per_block = atoi(argv[1]);
    const char* img_path = argv[3];
    double areaKm2 = 1.0;
    if (argc >= 5) {
        areaKm2 = atof(argv[4]);
    }

    unsigned long long base = total_samples_requested / (unsigned long long)world_size;
    unsigned long long remainder = total_samples_requested % (unsigned long long)world_size;
    unsigned long long samples_per_rank = base + ( (unsigned long long)world_rank < remainder ? 1ULL : 0ULL );

    if (world_rank == 0) {
        printf("MPI ranks: %d total requested samples: %llu\n", world_size, total_samples_requested);
        printf("Samples per rank (base=%llu, remainder=%llu)\n", base, remainder);
    }

    int width = 0, height = 0;
    unsigned char* h_img = nullptr;
    size_t img_size = 0;

    if (world_rank == 0) {

        printf("argv[0] = %s\n", argv[0]);
        printf("argv[1] = %s\n", argv[1]);
        printf("argv[2] = %s\n", argv[2]);
        printf("argv[3] = %s\n", argv[3]);
        printf("argv[4] = %s\n\n", argv[4]);


        h_img = load_png_rgb(img_path, &width, &height);
        if (!h_img) {
            fprintf(stderr, "Erro ao carregar imagem %s\n", img_path);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        img_size = (size_t)width * (size_t)height * 3u;
        printf("Imagem carregada: %s (w=%d h=%d) size=%zu bytes\n", img_path, width, height, img_size);
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank != 0) {
        img_size = (size_t)width * (size_t)height * 3u;
        h_img = (unsigned char*)malloc(img_size);
        if (!h_img) {
            fprintf(stderr, "Rank %d: erro de malloc para h_img\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(h_img, (int)img_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "Nenhuma GPU detectada no rank %d\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int device_id = world_rank % device_count;
    CUDA_CHECK(cudaSetDevice(device_id));

    unsigned char* d_img = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_img, img_size));
    CUDA_CHECK(cudaMemcpy(d_img, h_img, img_size, cudaMemcpyHostToDevice));

    unsigned long long* d_count = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(unsigned long long)));

    unsigned long long max_threads = (unsigned long long)threads_per_block * 65535ULL;
    unsigned long long threads_needed = samples_per_rank < max_threads ? samples_per_rank : max_threads;
    // garantir pelo menos 1 thread
    if (threads_needed == 0) threads_needed = 1ULL;

    int grid = (int)((threads_needed + threads_per_block - 1ULL) / threads_per_block);
    if (grid < 1) grid = 1;

    unsigned long long seed = (unsigned long long)time(NULL) ^ (unsigned long long)(world_rank * 0x9e3779b97f4a7c15ULL);

    if (world_rank == 0) {
        printf("Lançando kernel: grid=%d block=%d threads_needed=%llu samples_per_rank_max=%llu\n",
               grid, threads_per_block, threads_needed, samples_per_rank);
    }

    monte_kernel<<<grid, threads_per_block>>>(d_img, width, height, samples_per_rank, seed, d_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long long local_blue_count = 0;
    CUDA_CHECK(cudaMemcpy(&local_blue_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    unsigned long long global_red_count = 0;
    MPI_Reduce(&local_blue_count, &global_red_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    unsigned long long local_samples_done = samples_per_rank;
    unsigned long long global_samples_done = 0;
    MPI_Reduce(&local_samples_done, &global_samples_done, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        double fraction_red = 0.0;
        if (global_samples_done > 0) fraction_red = (double)global_red_count / (double)global_samples_done;
        unsigned long long total_pixels = (unsigned long long)width * (unsigned long long)height;
        double estimated_blue_pixels = fraction_red * (double)total_pixels;

        double areaSubArea = fraction_red * areaKm2;
        printf("\n");
        printf("Amostras totais = %llu\n", global_samples_done);
        printf("Amostras no pixel vermelho = %llu\n", global_red_count);
        printf("Área da subárea = %.2f km²\n", areaSubArea);
    }

    // cleanup
    CUDA_CHECK(cudaFree(d_img));
    CUDA_CHECK(cudaFree(d_count));
    free(h_img);

    MPI_Finalize();
    return 0;
}
