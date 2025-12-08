// To execute:
// mpirun -np 4 ./build/monte_mpi 100000 maui.png 100.0

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "png_loader.h"

static inline int is_red(unsigned char r, unsigned char g, unsigned char b) {
    return (r == 255 && g == 0 && b == 0);
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double start_time = MPI_Wtime();

    if (argc < 4) {
        if (rank == 0) {
            printf("Uso:\n");
            printf("  mpirun -np N ./monte_mpi total_samples caminho_imagem.png area_km2\n");
            printf("Exemplo:\n");
            printf("  mpirun -np 4 ./monte_mpi 10000000 maui.png 100.0\n");
        }
        MPI_Finalize();
        return 0;
    }

    unsigned long long total_samples = strtoull(argv[1], NULL, 10);
    const char* img_path = argv[2];
    double areaKm2 = atof(argv[3]);

    // dividir amostras entre ranks
    unsigned long long base = total_samples / size;
    unsigned long long remainder = total_samples % size;
    unsigned long long samples_per_rank = base + (rank < remainder ? 1ULL : 0ULL);

    if (rank == 0) {
        printf("argv[0] = %s\n", argv[0]);
        printf("argv[1] = %s\n", argv[1]);
        printf("argv[2] = %s\n", argv[2]);
        printf("argv[3] = %s\n\n", argv[3]);

        printf("MPI ranks: %d\n", size);
        printf("Total samples: %llu\n", total_samples);
        printf("Samples per rank (base=%llu, remainder=%llu)\n", base, remainder);
    }

    int width = 0, height = 0;
    unsigned char* full_img = NULL;
    size_t img_size = 0;

    if (rank == 0) {
        full_img = load_png_rgb(img_path, &width, &height);
        if (!full_img) {
            fprintf(stderr, "Erro: não foi possível abrir %s\n", img_path);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        img_size = (size_t)width * height * 3;
        printf("Imagem carregada: %dx%d, %zu bytes\n", width, height, img_size);
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        img_size = (size_t)width * height * 3;
        full_img = malloc(img_size);
    }

    MPI_Bcast(full_img, img_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    unsigned long long total_pixels = (unsigned long long)width * height;

    // RNG seed
    srand((unsigned int)time(NULL) + rank * 1234567);

    unsigned long long local_count = 0;

    for (unsigned long long i = 0; i < samples_per_rank; i++) {
        unsigned long long idx = ((unsigned long long)rand() * rand()) % total_pixels;

        unsigned long long basepix = idx * 3ULL;
        unsigned char r = full_img[basepix + 0];
        unsigned char g = full_img[basepix + 1];
        unsigned char b = full_img[basepix + 2];

        if (is_red(r, g, b))
            local_count++;
    }

    unsigned long long global_count = 0;
    unsigned long long global_samples = 0;

    MPI_Reduce(&local_count,   &global_count,   1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&samples_per_rank, &global_samples, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double fraction = (double)global_count / (double)global_samples;
        double area_est = fraction * areaKm2;

        printf("\nTotal samples = %llu\n", global_samples);
        printf("Hits in red pixels = %llu\n", global_count);
        printf("Fraction = %.6f\n", fraction);
        printf("Área estimada = %.3f km²\n", area_est);
    }

    free(full_img);

    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("\nTempo total de execução (MPI): %.6f segundos\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
