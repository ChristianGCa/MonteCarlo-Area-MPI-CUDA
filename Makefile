all: build/monte_mpi_cuda build/monte_mpi

build/monte_mpi_cuda: src/monte_mpi_cuda.cu src/png_loader.h
	@mkdir -p build
	nvcc -O3 -ccbin mpicxx src/monte_mpi_cuda.cu -o build/monte_mpi_cuda -lcurand -lpng

build/monte_mpi: src/monte_mpi.c src/png_loader.h
	@mkdir -p build
	mpicc -O3 src/monte_mpi.c -o build/monte_mpi -lpng

clean:
	rm -rf build