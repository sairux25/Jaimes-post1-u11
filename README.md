# Laboratorio Unidad 11 — Benchmark CUDA y Aceleradores de IA
Sara Jaimes Castañeda · Arquitectura de Computadores

Este repositorio contiene el post-contenido de la Unidad 11. La Parte 1 implementa en CUDA C la suma de vectores, la multiplicación de matrices con tiling en shared memory y una versión de la suma de vectores con streams y copia asíncrona, y compara los tiempos de CPU y GPU medidos con cudaEvent. La Parte 2 corresponde al análisis comparativo de aceleradores de IA, que se desarrolla en docs/analisis-comparativo.md.

Entorno utilizado

Windows 11 Home con WSL2 y Ubuntu 24.04.5 LTS. La GPU es una NVIDIA GeForce RTX 4050 Laptop con 6141 MiB de memoria, compute capability 8.9 y driver 610.78. Se usó CUDA Toolkit 12.8 con nvcc V12.8.93 y gcc 13.3.0.

Los programas se compilaron desde la raíz del repositorio con los siguientes comandos.

```
nvcc -O2 -arch=sm_89 -o vectorAdd src/vectorAdd.cu
nvcc -O2 -arch=sm_89 -DN=1048576 -o vectorAdd src/vectorAdd.cu
nvcc -O2 -arch=sm_89 -DN=4194304 -o vectorAdd src/vectorAdd.cu
nvcc -O2 -arch=sm_89 -o matMul src/matMul.cu
nvcc -O2 -arch=sm_89 -o vectorAddStream src/vectorAddStream.cu
```

# PARTE 1 — Benchmark CUDA CPU vs GPU

## Checkpoint 1 — Benchmark vectorAdd
Capturas: capturas/CP1_compilacion_vectorAdd.png y capturas/CP1_ejecucion_vectorAdd.png

La primera vez que se compiló solo con nvcc -O2, el kernel tardó 584.96 ms, un tiempo muy alto para sumar 16M elementos. Al no indicar la arquitectura, nvcc genera un código genérico y el driver lo traduce para la GPU que lo ejecuta justo en esa primera corrida, así que ese tiempo terminó dentro de la medición. Para evitarlo se agregó -arch=sm_89 al comando, que corresponde a la GPU de este equipo. Con ese cambio la CPU tardó 42.21 ms, el kernel 7.24 ms y la salida mostró Errores: 0.

## Checkpoint 2 — Multiplicación de matrices con tiling
Capturas: capturas/CP2_compilacion_matMul.png y capturas/CP2_ejecucion_matMul.png

Del programa de matMul solo estaba el kernel con tiling, así que se escribió una versión naive que lee todo desde memoria global, la multiplicación en CPU para comparar y un main que mide los dos kernels con cudaEvent en N=512 y N=1024. Cada kernel se lanza una vez antes de medir para que la carga inicial no se cuente en ninguno. Los resultados coinciden con la CPU, con un error máximo de 4.58e-05 en N=512 y de 9.16e-05 en N=1024, muy por debajo de 1e-3.

| N | Naive (ms) | Tiling (ms) | Speedup |
|---|---|---|---|
| 512 | 0.618 | 0.491 | 1.26x |
| 1024 | 4.252 | 3.363 | 1.26x |
