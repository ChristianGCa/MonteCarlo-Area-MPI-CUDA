# Monte Carlo MPI + CUDA — Contagem de Pixels Vermelhos em PNG

Este projeto implementa um algoritmo de Monte Carlo para estimar a quantidade de pixels vermelhos (RGB = 255, 0, 0) em uma imagem PNG usando:
- MPI para paralelização entre múltiplos processos
- CUDA para paralelização na GPU
- Um loader PNG simples (png_loader.h) implementado manualmente

Ademais, o objetivo principal é calcular a área aproximada de lagos, terrenos ou outras formas com muitas imperfeições usando um mapa como imagem de entrada.
Para isso, obtivemos prints retangulares de áreas do Google Earth e, usando a ferramenta de medição do próprio, desenhamos um polígono retangular para saber a área da região do print.
Desse modo, podemos colorir de vermelho alguma forma no mapa (um lago, por exemplo) usando o Paint (ferramenta do Windows) e usar o algoritmo de Monte Carlo implementado para predizer o tamanho dela.

## Compilando e executando

Para compilar:
```bash
make
```

Para executar a versão MPI:
```bash
mpirun -np <"processos paralelos"> ./build/monte_mpi <"número de amostras"> <"imagem"> <"Área da imagem (Km²)">
```
Exemplo:
```bash
mpirun -np 4 ./build/monte_mpi 100000 test_12.png 100.0
```

Para executar a versão MPI+CUDA:
```bash
mpirun -np <"processos paralelos"> ./build/monte_mpi_cuda <"tamanho do bloco CUDA"> <"número de amostras"> <"imagem"> <"Área da imagem (Km²)">
```
Exemplo:
```bash
mpirun -np 4 ./build/monte_mpi_cuda 256 10000000 test_12.png 100.0
```
